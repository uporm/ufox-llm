/// 演示 Phase 4 记忆系统：
/// 1. 使用 InMemory 后端写入会话记忆 / 用户偏好
/// 2. 新会话通过 enable_perceive 自动检索记忆
/// 3. 也展示 SqliteMemory 的跨进程持久化路径
use anyhow::Result;
use ufox_arc::{Agent, ArcError, InMemoryStore, MemoryScope, SqliteMemory};
use ufox_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // ---- 场景 A：InMemory 后端，同进程跨会话 ----
    println!("=== 场景 A：InMemory 跨会话 ===");
    in_memory_demo().await?;

    // ---- 场景 B：SQLite 后端（持久化文件） ----
    println!("\n=== 场景 B：SQLite 持久化 ===");
    sqlite_demo().await?;

    Ok(())
}

async fn in_memory_demo() -> Result<()> {
    let store = InMemoryStore::new();
    let llm = Client::from_env()?;

    let agent = Agent::builder()
        .llm(llm)
        .instructions("你是一个助手。如果上下文中有 [Memory Context] 块，请优先使用其中的信息。")
        .memory(store)
        .enable_perceive(true)
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // 第一个线程：用户分享偏好，手动写入用户记忆
    let thread1 = agent.thread("alice", "session-001");
    let memory = agent.memory().map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    memory
        .remember_user(
            "alice",
            "用户偏好：回答请使用简洁的要点格式，不要长篇大论",
            vec!["preference".into()],
        )
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    memory
        .remember_thread(
            thread1.thread_id.0.as_str(),
            "本次会话主题：Rust async 最佳实践",
            vec!["topic".into()],
        )
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    println!("[session-001] 已写入 1 条用户偏好 + 1 条会话记忆");

    // 查看写入结果
    let user_mems = memory
        .user_memories("alice")
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;
    let session_mems = memory
        .thread_memories(thread1.thread_id.0.as_str())
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    println!(
        "  用户记忆: {} 条, 会话记忆: {} 条",
        user_mems.len(),
        session_mems.len()
    );

    // 第二个会话（同一用户）：开启 Perceive，应能看到用户偏好
    let thread2 = agent.thread("alice", "session-002");

    let result = agent
        .run(&thread2, "用一句话解释什么是 tokio？")
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    println!("[session-002] 回复：{}", result.response.text);

    // 打印 Perceive 步骤的命中条数
    for step in &result.trace.steps {
        if matches!(step.kind, ufox_arc::StepKind::Perceive)
            && let ufox_arc::StepOutput::MemoryHits(ref hits) = step.output
        {
            println!("[session-002] Perceive 命中 {} 条记忆", hits.len());
        }
    }

    Ok(())
}

async fn sqlite_demo() -> Result<()> {
    let db_path = "/tmp/ufox_memory_demo.db";
    let store = SqliteMemory::open(db_path)
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    let llm = Client::from_env()?;

    let agent = Agent::builder()
        .llm(llm)
        .instructions("你是一个助手。")
        .memory(store)
        .enable_perceive(true)
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let thread = agent.thread("bob", "sqlite-session-001");
    let memory = agent.memory().map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    // 写入一条长期用户记忆
    memory
        .remember_user(
            "bob",
            "bob 是一名后端工程师，熟悉 Go 和 Rust，正在学习 AI 应用开发",
            vec!["profile".into()],
        )
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    println!("[sqlite] 已写入用户档案到 {db_path}");

    let mems = memory
        .user_memories(thread.user_id.0.as_str())
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;
    for m in &mems {
        let scope_label = match &m.scope {
            MemoryScope::User { user_id } => format!("user:{}", user_id),
            MemoryScope::Thread { thread_id } => format!("thread:{}", thread_id),
        };
        println!("  [{}] {}", scope_label, m.content);
    }

    Ok(())
}
