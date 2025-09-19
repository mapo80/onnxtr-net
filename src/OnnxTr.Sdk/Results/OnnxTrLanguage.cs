using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Describes a language prediction for a page.
/// </summary>
public sealed class OnnxTrLanguage
{
    [JsonPropertyName("value")]
    public string? Value { get; init; }

    [JsonPropertyName("confidence")]
    public float Confidence { get; init; }
}
