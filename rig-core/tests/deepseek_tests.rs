use rig::providers::deepseek::{ClientBuilder, DEEPSEEK_CHAT};
use rig::completion::{CompletionRequest, ModelChoice};

#[tokio::test]
async fn test_deepseek_completion() {
    // Replace with your DeepSeek API key or set the DEEPSEEK_API_KEY environment variable
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set");

    let client = ClientBuilder::new(&api_key).build();
    let model = client.completion_model(DEEPSEEK_CHAT);

    let request = CompletionRequest::from_prompt("Hello, who are you?")
        .with_max_tokens(50)
        .with_temperature(0.5);

    let response = model.completion(request).await.expect("Failed to get completion");

    if let ModelChoice::Message(content) = response.choice {
        println!("DeepSeek response: {}", content);
        assert!(!content.is_empty(), "Response content is empty");
    } else {
        panic!("Unexpected response choice");
    }
}
