use crate::memory::{Memory, MemoryFilter, MemoryScope, MemoryStore};
use crate::thread::{ThreadId, UserId};

/// 检索上下文记忆：先取线程记忆，再补充用户记忆，按时间倒序合并。
///
/// `limit` 为总条数上限（线程与用户各占一半，不足时自动补另一半）。
pub async fn retrieve_context(
    store: &dyn MemoryStore,
    thread_id: &ThreadId,
    user_id: &UserId,
    limit: usize,
) -> Vec<Memory> {
    let half = (limit / 2).max(1);

    let thread_hits = store
        .find(MemoryFilter {
            scope: Some(MemoryScope::Thread {
                thread_id: thread_id.clone(),
            }),
            limit: Some(half),
            ..Default::default()
        })
        .await
        .unwrap_or_default();

    let user_hits = store
        .find(MemoryFilter {
            scope: Some(MemoryScope::User {
                user_id: user_id.clone(),
            }),
            limit: Some(limit.saturating_sub(thread_hits.len()).max(1)),
            ..Default::default()
        })
        .await
        .unwrap_or_default();

    let mut all = thread_hits;
    all.extend(user_hits);
    all
}

/// 将记忆条目列表格式化为注入消息的文本块。
pub fn format_context(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return String::new();
    }
    let mut out = String::from("[Memory Context]\n");
    for m in memories {
        out.push_str("- ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out
}
