/// 演示 Phase 5 多模态输入：
/// 1. 附加图片 URL 并提问
/// 2. 附加本地文本文件（文档模态），后续追问无需重传
/// 3. 基于同一 thread 继续追问，无需重复上传附件
use anyhow::Result;
use std::io::Write as _;
use ufox_arc::{Agent, ArcError, InMemoryStore, Modality};
use ufox_llm::{Client, MediaSource};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let llm = Client::from_env()?;
    let store = InMemoryStore::new();

    let agent = Agent::builder()
        .llm(llm)
        .instructions("你是一个多模态助手，可以理解图片和文档内容。请用中文回答。")
        .memory(store)
        .enable_perceive(true)
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let thread = agent.thread("alice", "mm-session-001");

    // ---- 场景 A：附加图片 URL，然后提问 ----
    println!("=== 场景 A：图片理解 ===");
    let img_ref = thread
        .attach(
            MediaSource::Url {
                url: "https://fastly.picsum.photos/id/294/800/600.jpg?hmac=X4RiVynizog5zMK1YZqNYt7sT1XJVHx4bRv9ZDCpPwI".into(),
            },
            Modality::Image,
            vec!["image".into()],
        )
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;
    println!("图片已附加，MediaRef: {}", img_ref.0);

    let result = agent
        .run(&thread, "请描述一下刚才附加的图片的内容。")
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;
    println!("回复：{}", result.response.text);

    // ---- 场景 B：附加本地文本文件，后续追问 ----
    println!("\n=== 场景 B：文档理解（不重复上传）===");

    // 创建临时文档
    let mut tmp = tempfile::NamedTempFile::new()?;
    writeln!(
        tmp,
        "项目名称：ufox-arc\n\
         版本：0.1.0\n\
         描述：基于 ufox-llm 的轻量 AI Agent 运行时，支持工具调用、记忆系统和多模态输入。\n\
         主要特性：\n\
         1. 执行轨迹记录\n\
         2. 工具注册与调用\n\
         3. 会话级/用户级记忆\n\
         4. 多模态附件支持"
    )?;

    let doc_ref = thread
        .attach(
            MediaSource::File {
                path: tmp.path().to_path_buf(),
            },
            Modality::Document,
            vec!["document".into(), "ufox".into()],
        )
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;
    println!("文档已附加，MediaRef: {}", doc_ref.0);

    // 第一个追问
    let r1 = agent
        .run(&thread, "这份文档描述的是什么项目？")
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;
    println!("追问 1：{}", r1.response.text);

    // 第二个追问，无需重新上传文档
    let r2 = agent
        .run(&thread, "该项目有哪些主要特性？请列出要点。")
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;
    println!("追问 2：{}", r2.response.text);

    Ok(())
}
