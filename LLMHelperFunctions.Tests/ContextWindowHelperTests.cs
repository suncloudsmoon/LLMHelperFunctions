namespace LLMHelperFunctions.Tests
{
    public class ContextWindowHelperTests
    {
        [Fact]
        public void TokenTest()
        {
            string content = "Sample hello";
            int tokenCount = ContextWindowHelper.CharToTokenCount(content.Count());
            int charCount = ContextWindowHelper.TokenToCharCount(tokenCount);
            Assert.Equal(charCount, content.Count());
        }
    }
}