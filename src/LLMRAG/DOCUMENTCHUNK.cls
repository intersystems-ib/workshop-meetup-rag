Class LLMRAG.DOCUMENTCHUNK Extends %Persistent
{
    Property Document As %String(MAXLEN = 256);

    Property Phrase As %String(MAXLEN = 1000);

    Property VectorizedPhrase As %Library.Vector(DATATYPE = "DECIMAL", LEN = 384);
}