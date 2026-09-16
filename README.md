**Studying Symbolic/Discrete Language Modeling**

A symbolic language model may be fully interpretable, with embeddings/residuals consisting of human understandable categorical values, and with discrete transformation rules operating on these embeddings/residuals.

Current LLMs rely on high dimensional continual embeddings/residuals and complex transformations such as attention (which factors in every single previous residual at the current layer). Studies have been done to interpret residuals by training additional networks on those residuals to isolate codes and so on. Recent findings indicate that this is non-trivial due to LLMs embeddings/residuals having a high degree of superposition.

A symbolic language model was probably intractable to build previously using discrete optimization methods. It is now tractable to build thanks to modern LLMs as smart optimizers that can make targeted edits to structured symbolic knowledge representations.

As the ai-2027.com writeup says, eventually “Like a software engineer simplifying spaghetti code into a few elegant lines of Python, [Agent-4, an advanced LLM] untangles its own circuits into something sensible and rational. The new AI is somewhere between a neural net and a traditional computer program, with much of its weights rewritten in readable (albeit very long and arcane) code.“


I have set up a looping structure to optimize a symbolic language model by repeatedly having Claude (or Grok) work on it. The structure is based on a classic Transformer:

Token IDs are mapped to categorical embeddings from a lookup table (for example, the token for "cat" may map to the embedding {"is_animal": true, "word_part_of_speech": "noun", ... })

Embeddings are transformed (to residuals) from an attention approximation and a FFN approximation.

Attention approximation: attention heads can retrieve first matching token, last matching token, a fixed offset, or most likely/average (Q/K approximation) and then transform the embedding/residual based on the result.

FFN approximation: the residual may be transformed based on itself.

Unembedding to predict next token (rules based).

https://manifund.org/projects/discrete-language-modeling
