use reqwest::header::HeaderMap;
use super::completion::{
    // Re-export or reference your completion constants
    DEEPSEEK_API_BASE_URL,
    // And the CompletionModel struct
    CompletionModel
};

/// A builder for the DeepSeek Client (similar to Anthropic's `ClientBuilder`).
#[derive(Clone)]
pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    betas: Option<Vec<&'a str>>,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: DEEPSEEK_API_BASE_URL, // "https://api.deepseek.com"
            betas: None,
        }
    }

    pub fn base_url(mut self, url: &'a str) -> Self {
        self.base_url = url;
        self
    }

    pub fn beta(mut self, beta: &'a str) -> Self {
        if let Some(ref mut betas) = self.betas {
            betas.push(beta);
        } else {
            self.betas = Some(vec![beta]);
        }
        self
    }

    pub fn build(self) -> Client {
        Client::new(self.api_key, self.base_url, self.betas)
    }
}

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str, base_url: &str, betas: Option<Vec<&str>>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse().unwrap());
        // If you want to replicate an "x-deepseek-beta" header:
        if let Some(betas) = betas {
            headers.insert("x-deepseek-beta", betas.join(",").parse().unwrap());
        }

        let http_client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .expect("DeepSeek reqwest client build failed");

        Self {
            base_url: base_url.to_string(),
            http_client,
        }
    }

    /// Like Anthropic's `.completion_model(...)`
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Example method for making requests if you want a general “post”
    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        // Combine base_url + path, watch out for extra slashes
        let url = format!("{}/{}", self.base_url.trim_end_matches('/'), path.trim_start_matches('/'));
        self.http_client.post(url)
    }
}
