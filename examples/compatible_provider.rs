use ufox_llm::{Client, Provider};

fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::Compatible)
        .base_url("https://api.deepseek.com/v1")
        .api_key("sk-xxx")
        .model("deepseek-chat")
        .build()?;

    assert_eq!(client.base_url(), "https://api.deepseek.com/v1");
    Ok(())
}
