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
    /// <summary>
    /// Provides functionality to interact with an Ollama API endpoint for retrieving model details.
    /// </summary>
    public class Ollama
    {
        private Uri _endpoint;
        private HttpClient _httpClient;
        private static readonly Uri _showApiRelativeUri = new Uri("api/show", UriKind.Relative);

        /// <summary>
        /// Initializes a new instance of the <see cref="Ollama"/> class with the specified API endpoint.
        /// </summary>
        /// <param name="endpoint">
        /// The base URI of the Ollama API endpoint.
        /// </param>
        public Ollama(Uri endpoint)
        {
            _endpoint = endpoint;
            _httpClient = new HttpClient();
        }

        /// <summary>
        /// Sends a request to the Ollama API to retrieve details about the specified model.
        /// </summary>
        /// <param name="modelName">The name of the model for which details are requested.</param>
        /// <param name="verbose">
        /// A value indicating whether to request verbose information. The default is <c>false</c>.
        /// </param>
        /// <returns>
        /// A task that represents the asynchronous operation. The task result contains a dictionary mapping
        /// keys to objects representing the model details returned by the API.
        /// </returns>
        /// <exception cref="HttpRequestException">
        /// Thrown when the HTTP request fails.
        /// </exception>
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

        /// <summary>
        /// Determines whether the provided <see cref="OpenAIModelCollection"/> indicates that the model provider is Ollama.
        /// </summary>
        /// <param name="availableModels">
        /// A collection of OpenAI models. The first model's <see cref="OpenAI.Models.OpenAIModel.OwnedBy"/> property is used to determine
        /// the model provider.
        /// </param>
        /// <returns>
        /// <c>true</c> if the model provider is detected as Ollama; otherwise, <c>false</c>.
        /// </returns>
        /// <exception cref="OllamaException">
        /// Thrown if the <paramref name="availableModels"/> collection is empty, meaning that no models are available to inspect.
        /// </exception>
        public static bool IsOllama(OpenAIModelCollection availableModels)
        {
            return (availableModels.Count == 0) ?
                throw new OllamaException("Unable to detect model provider as Ollama due to 0 available models!") : availableModels.First().OwnedBy == "library";
        }
    }

    /// <summary>
    /// Represents errors that occur when interacting with the Ollama API.
    /// </summary>
    public class OllamaException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="OllamaException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The message that describes the error.</param>
        public OllamaException(string message) : base(message) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="OllamaException"/> class with a specified error message and a reference
        /// to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">
        /// The exception that is the cause of the current exception, or a null reference if no inner exception is specified.
        /// </param>
        public OllamaException(string message, Exception innerException) : base(message, innerException) { }
    }
}