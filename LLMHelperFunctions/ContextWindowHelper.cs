using System;
using System.Collections.Generic;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Threading.Tasks;
using static LLMHelperFunctions.ContextWindowHelper;
using System.Text.RegularExpressions;

namespace LLMHelperFunctions
{
    /// <summary>
    /// Provides helper functions for determining the context window size (i.e. number of tokens)
    /// for embedding and conversational models from different providers such as OpenAI and Ollama.
    /// </summary>
    public class ContextWindowHelper
    {
        /// <summary>
        /// Identifiers for supported model providers.
        /// </summary>
        public enum ModelProvider
        {
            /// <summary>
            /// OpenAI models.
            /// </summary>
            OpenAI,
            /// <summary>
            /// Ollama models.
            /// </summary>
            Ollama
        };

        private static readonly Dictionary<string, int> _openAIModelContextWindows = new Dictionary<string, int>()
        {
            // GPT-4o
            { "gpt-4o-2024-11-20", 128000 },
            { "gpt-4o-2024-08-06", 128000 },
            { "gpt-4o-2024-05-13", 128000 },

            // GPT-4o mini
            { "gpt-4o-mini-2024-07-18", 128000 },

            // o1 and o1-mini
            { "o1-2024-12-17", 200000 },
            { "o1-mini-2024-09-12", 128000 },
            { "o1-preview-2024-09-12", 128000 },

            // o3-mini
            { "o3-mini-2025-01-31", 200000 },

            // GPT-4o and GPT-4o-mini Realtime
            { "gpt-4o-realtime-preview-2024-12-17", 128000 },
            { "gpt-4o-realtime-preview-2024-10-01", 128000 },
            { "gpt-4o-mini-realtime-preview-2024-12-17", 128000 },

            // GPT-4o and GPT-4o-mini Audio
            { "gpt-4o-audio-preview-2024-12-17", 128000 },
            { "gpt-4o-audio-preview-2024-10-01", 128000 },
            { "gpt-4o-mini-audio-preview-2024-12-17", 128000 },

            // GPT-4 Turbo and GPT-4
            { "gpt-4-turbo-2024-04-09", 128000 },
            { "gpt-4-0125-preview", 128000 },
            { "gpt-4-1106-preview", 128000 },
            { "gpt-4-0613", 8192 },
            { "gpt-4-0314", 8192 },

            // GPT-3.5 Turbo
            { "gpt-3.5-turbo-0125", 16385 },
            { "gpt-3.5-turbo-1106", 16385 },
            { "gpt-3.5-turbo-instruct", 4096 },

            // Embeddings
            { "text-embedding-3-large", 8191 },
            { "text-embedding-3-small", 8191 },
            { "text-embedding-ada-002", 8191 }
        };

        private static readonly Dictionary<string, int> _openAIModelAliases = new Dictionary<string, int>()
        {
            { "gpt-4o", _openAIModelContextWindows["gpt-4o-2024-08-06"] },
            { "chatgpt-4o-latest", _openAIModelContextWindows["gpt-4o-2024-08-06"] },
            { "gpt-4o-mini", _openAIModelContextWindows["gpt-4o-mini-2024-07-18"] },
            { "o1", _openAIModelContextWindows["o1-2024-12-17"] },
            { "o1-mini", _openAIModelContextWindows["o1-mini-2024-09-12"] },
            { "o3-mini", _openAIModelContextWindows["o3-mini-2025-01-31"] },
            { "o1-preview", _openAIModelContextWindows["o1-preview-2024-09-12"] },
            { "gpt-4o-realtime-preview", _openAIModelContextWindows["gpt-4o-realtime-preview-2024-12-17"] },
            { "gpt-4o-mini-realtime-preview", _openAIModelContextWindows["gpt-4o-mini-realtime-preview-2024-12-17"] },
            { "gpt-4o-audio-preview", _openAIModelContextWindows["gpt-4o-audio-preview-2024-12-17"] },
            
            { "gpt-4-turbo", _openAIModelContextWindows["gpt-4-turbo-2024-04-09"] },
            { "gpt-4-turbo-preview", _openAIModelContextWindows["gpt-4-0125-preview"] },
            { "gpt-4", _openAIModelContextWindows["gpt-4-0613"] },

            { "gpt-3.5-turbo", _openAIModelContextWindows["gpt-3.5-turbo-0125"] }
        };

        private static ContextLenCacheSystem _contextLenCacheSystem = new ContextLenCacheSystem(); // because network requests are slower

        // Helper functions for getting the context length for embedding models for different model providers (i.e. Ollama)

        /// <summary>
        /// Retrieves the context window (number of tokens) for the specified model from the given provider.
        /// </summary>
        /// <param name="provider">The model provider (e.g. <see cref="ModelProvider.OpenAI"/> or <see cref="ModelProvider.Ollama"/>).</param>
        /// <param name="endpoint">
        /// The endpoint URI to use when accessing the model information. For example, this can be used by
        /// the Ollama provider to fetch details from a remote server.
        /// </param>
        /// <param name="model">The identifier or alias for the model whose context window is requested.</param>
        /// <returns>
        /// A task that represents the asynchronous operation. The task result contains the context window size as an integer.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// Thrown when the specified model does not have a known context length (for OpenAI).
        /// </exception>
        /// <exception cref="OllamaException">
        /// Thrown when the context window cannot be retrieved from an Ollama provider.
        /// </exception>
        /// <exception cref="NotImplementedException">
        /// Thrown when the specified model provider is not supported.
        /// </exception>
        public static async Task<int> GetContextWindow(ModelProvider provider, Uri endpoint, string model)
        {
            // Cache system
            int contextWindow;
            if (_contextLenCacheSystem.TryGetContextWindow(provider, model, out contextWindow))
                return contextWindow;

            switch (provider)
            {
                case ModelProvider.OpenAI:
                    if (_openAIModelAliases.TryGetValue(model, out contextWindow))
                    {
                        _contextLenCacheSystem.Cache(provider, model, contextWindow);
                        return contextWindow;
                    }
                    else if (_openAIModelContextWindows.TryGetValue(model, out contextWindow))
                    {
                        _contextLenCacheSystem.Cache(provider, model, contextWindow);
                        return contextWindow;
                    }
                    else
                    {
                        throw new ArgumentException($"The model {model} does not have a known context length!");
                    }
                case ModelProvider.Ollama:
                    Ollama ollamaInstance = new Ollama(endpoint);
                    var dict = await ollamaInstance.Show(model, verbose: true);
                    
                    // Check if Ollama has manually overridden the num_ctx parameter
                    int? num_ctx = ExtractNumCtx(dict);
                    if (num_ctx != null)
                        return (int) num_ctx;

                    // Navigate to "model_info"
                    if (dict.TryGetValue("model_info", out var modelInfoObj) && modelInfoObj is JsonElement modelInfoElement)
                    {
                        // Use JsonNode or JsonElement to search for "context_length" key
                        var modelInfoNode = JsonNode.Parse(modelInfoElement.GetRawText());

                        foreach (var keyValuePair in modelInfoNode.AsObject())
                        {
                            // Search for a nested object containing "context_length"
                            if (keyValuePair.Key.EndsWith(".context_length"))
                            {
                                int ctx = int.Parse(keyValuePair.Value.ToString());
                                _contextLenCacheSystem.Cache(provider, model, ctx);
                                return ctx;
                            }
                        }
                    }

                    throw new OllamaException("Unable to fetch context window from Ollama!");
                default:
                    throw new NotImplementedException("The model provider requested cannot be found!");
            }
        }

        /// <summary>
        /// Splits the provided content into multiple chunks, each corresponding to an approximate number of tokens.
        /// This method estimates the number of characters to include per chunk based on the provided token count.
        /// </summary>
        /// <param name="content">The text content to be split into chunks.</param>
        /// <param name="numTokens">
        /// The target number of tokens per chunk. The method converts this to an estimated character count.
        /// </param>
        /// <returns>
        /// An enumerable of string chunks created from the content.
        /// </returns>
        public static IEnumerable<string> Chunkify(string content, int numTokens)
        {
            int charCount = TokenToCharCount(numTokens);
            List<string> paragraphs = new List<string>();

            for (int i = 0; i < content.Length; i += charCount)
            {
                if (i + charCount > content.Length)
                    charCount = content.Length - i;

                paragraphs.Add(content.Substring(i, charCount));
            }
            return paragraphs;
        }

        /// <summary>
        /// Converts a given character count to an estimated token count.
        /// This estimation is based on a simple heuristic (approximately 1 token per 4 characters)
        /// as used by OpenAI's tokenizer.
        /// </summary>
        /// <param name="charCount">The number of characters.</param>
        /// <returns>The estimated number of tokens.</returns>
        public static int CharToTokenCount(int charCount)
        {
            // https://platform.openai.com/tokenizer
            return charCount / 4;
        }

        /// <summary>
        /// Converts a given token count to an estimated character count.
        /// This estimation is based on a simple heuristic (approximately 4 characters per token)
        /// as used by OpenAI's tokenizer.
        /// </summary>
        /// <param name="tokenCount">The number of tokens.</param>
        /// <returns>The estimated number of characters.</returns>
        public static int TokenToCharCount(int tokenCount)
        {
            // https://platform.openai.com/tokenizer
            return tokenCount * 4;
        }

        private static int? ExtractNumCtx(Dictionary<string, object> apiResponse)
        {
            if (apiResponse == null)
            {
                throw new ArgumentNullException(nameof(apiResponse));
            }

            // Check if the response contains a "modelfile" key.
            if (apiResponse.TryGetValue("modelfile", out var modelfileObj) && modelfileObj != null)
            {
                string modelfile = modelfileObj.ToString();
                if (!string.IsNullOrWhiteSpace(modelfile))
                {
                    // Regex pattern looks for a line that starts with "PARAMETER num_ctx" followed by a number.
                    var regex = new Regex(@"PARAMETER\s+num_ctx\s+(\d+)", RegexOptions.IgnoreCase);
                    var match = regex.Match(modelfile);
                    if (match.Success && match.Groups.Count > 1)
                    {
                        if (int.TryParse(match.Groups[1].Value, out int numCtx))
                        {
                            return numCtx;
                        }
                    }
                }
            }

            // Return null if num_ctx is not found or could not be parsed.
            return null;
        }
    }


    internal class ContextLenCacheSystem 
    {
        private static Dictionary<ModelProvider, Dictionary<string, int>> _modelContextLenCache = new Dictionary<ModelProvider, Dictionary<string, int>>();
        
        public void Cache(ModelProvider provider, string model, int contextWindow)
        {
            CheckModelProviderValidity(provider);
            _modelContextLenCache[provider][model] = contextWindow;
        }

        public bool TryGetContextWindow(ModelProvider provider, string model, out int contextWindow)
        {
            contextWindow = 0;
            CheckModelProviderValidity(provider);

            Dictionary<string, int> ctxDict;
            if (!_modelContextLenCache.TryGetValue(provider, out ctxDict))
                return false;

            return ctxDict.TryGetValue(model,out contextWindow);
        }

        private static void CheckModelProviderValidity(ModelProvider provider)
        {
            switch (provider)
            {
                case ModelProvider.Ollama:
                case ModelProvider.OpenAI:
                    break;
                default:
                    throw new NotImplementedException($"Unknown model provider #{provider}!");
            }
        }
    }
}
