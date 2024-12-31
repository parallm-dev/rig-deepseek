#![doc = "DeepSeek API client and Rig integration"]

/// # Example
/// ```
/// use rig_core::providers::deepseek;
///
/// let client = deepseek::ClientBuilder::new("YOUR_DEEPSEEK_API_KEY").build();
/// let completion_model = client.completion_model("deepseek-chat");
/// ```

pub mod client;
pub mod completion;

pub use client::{Client, ClientBuilder};
pub use completion::{
    // Re-export your public items:
    CompletionModel,
    DEEPSEEK_CHAT,
    DEEPSEEK_CODER,
    // etc...
};
