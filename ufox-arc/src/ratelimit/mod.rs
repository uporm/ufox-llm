use std::time::{Duration, Instant};

use tokio::sync::Mutex;

/// 令牌桶速率限制器；线程安全，可通过 `Arc` 共享。
///
/// 每次 `acquire()` 消耗一个令牌。桶为空时休眠直到有令牌可用，
/// 不丢弃请求。
pub struct RateLimiter {
    inner: Mutex<Bucket>,
}

struct Bucket {
    /// 当前令牌数（可以是小数）。
    tokens: f64,
    /// 桶容量（突发上限）。
    capacity: f64,
    /// 每秒补充的令牌数。
    refill_per_sec: f64,
    last_refill: Instant,
}

impl RateLimiter {
    /// 创建速率限制器。`requests_per_second` 同时作为容量和补充速率。
    pub fn new(requests_per_second: f64) -> Self {
        assert!(
            requests_per_second > 0.0,
            "requests_per_second must be positive"
        );
        Self {
            inner: Mutex::new(Bucket {
                tokens: requests_per_second,
                capacity: requests_per_second,
                refill_per_sec: requests_per_second,
                last_refill: Instant::now(),
            }),
        }
    }

    /// 获取一个令牌；桶为空时阻塞等待。
    pub async fn acquire(&self) {
        loop {
            let wait = {
                let mut b = self.inner.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(b.last_refill).as_secs_f64();
                b.tokens = (b.tokens + elapsed * b.refill_per_sec).min(b.capacity);
                b.last_refill = now;

                if b.tokens >= 1.0 {
                    b.tokens -= 1.0;
                    None
                } else {
                    let deficit = 1.0 - b.tokens;
                    Some(Duration::from_secs_f64(deficit / b.refill_per_sec))
                }
            };
            match wait {
                None => break,
                Some(d) => {
                    tracing::debug!(wait_ms = d.as_millis(), "rate_limiter: throttling");
                    tokio::time::sleep(d).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn acquire_under_capacity_is_instant() {
        let rl = RateLimiter::new(10.0);
        let start = Instant::now();
        // 连续获取 5 个令牌（容量 10），应立即完成
        for _ in 0..5 {
            rl.acquire().await;
        }
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn acquire_over_capacity_throttles() {
        let rl = RateLimiter::new(100.0); // 100 rps，每令牌 10ms
        // 消耗全部令牌
        for _ in 0..100 {
            rl.acquire().await;
        }
        let start = Instant::now();
        rl.acquire().await; // 第 101 个必须等待
        // 应该等待约 10ms，给 50ms 宽裕
        assert!(start.elapsed() >= Duration::from_millis(5));
    }
}
