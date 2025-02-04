using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using OpenAI.Models;

namespace LLMHelperFunctions
{
    public class Ollama
    {
        private Uri _endpoint;
        private HttpClient _httpClient;
        private static readonly Uri _showApiRelativeUri = new Uri("api/show", UriKind.Relative);

        public Ollama(Uri endpoint)
        {
            _endpoint = endpoint;
            _httpClient = new HttpClient();
        }

        public async Task<Dictionary<string, object>> Show(string modelName, bool verbose = false)
        {
            Uri fullUri = new Uri(_endpoint, _showApiRelativeUri);

            // Create the request payload
            var payload = new
            {
                name = modelName,
                verbose = verbose
            };

            // Serialize the payload to JSON using System.Text.Json
            var jsonPayload = JsonSerializer.Serialize(payload);

            // Create a StringContent object to send the payload
            var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            // Send the POST request
            var response = await _httpClient.PostAsync(fullUri.OriginalString, content);
            response.EnsureSuccessStatusCode();

            // Read the response content as a string
            var responseString = await response.Content.ReadAsStringAsync();

            // Deserialize the JSON response into a Dictionary
            var responseData = JsonSerializer.Deserialize<Dictionary<string, object>>(responseString);

            return responseData;
        }

        public static bool IsOllama(OpenAIModelCollection availableModels)
        {
            return (availableModels.Count == 0) ?
                throw new OllamaException("Unable to detect model provider as Ollama due to 0 available models!") : availableModels.First().OwnedBy == "library";
        }
    }

    public class OllamaException : Exception
    {
        public OllamaException(string message) : base(message) { }
        public OllamaException(string message, Exception innerException) : base(message, innerException) { }
    }
}