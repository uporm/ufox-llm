use crate::memory::{Memory, MemoryFilter, MemoryScope, MemoryStore};
use crate::session::{SessionId, UserId};

/// 检索上下文记忆：先取会话记忆，再补充用户记忆，按时间倒序合并。
///
/// `limit` 为总条数上限（会话与用户各占一半，不足时自动补另一半）。
pub async fn retrieve_context(
    store: &dyn MemoryStore,
    session_id: &SessionId,
    user_id: &UserId,
    limit: usize,
) -> Vec<Memory> {
    let half = (limit / 2).max(1);

    let session_hits = store
        .find(MemoryFilter {
            scope: Some(MemoryScope::Session {
                session_id: session_id.clone(),
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
            limit: Some(limit.saturating_sub(session_hits.len()).max(1)),
            ..Default::default()
        })
        .await
        .unwrap_or_default();

    let mut all = session_hits;
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
