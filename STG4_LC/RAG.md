![RAG_User_Flow](rag.png)

# Query Translation
RAG工作流的第一阶段，接收用户输入的问题，以某种方式将其转化为提高检索效果的形式。因为用户的问题可能模棱两可，质量良莠不齐，我们在执行向量检索时不能召回很好的结果。

## Query Decomposition
查询分解中通常有这样三种方式进行查询转换，事实上往往会综合使用：
- Step-back question：将问题转换为更高层级的更抽象的问题。

    **通过将问题回退到一个更高层级的抽象问题，可以扩大检索范围，增加召回相关文档的可能性；在检索到相关文档后，可以进一步细化查询，找到更具体的答案。**

    **首先采用few_shot暗示大模型根据原始问题返回step—back问题，再将step-back问题retrieval得到的context和原始问题retrieval得到的context都融入prompt，合起来形成整体的context，然后再提出原始问题，请求回答。**

- Re-written：将问题同级重写（RAG-Fusion，Multi-Query）。

    **经典的Shotgun Approach核心思想就是通过​​生成大量可能的查询变体​​，覆盖用户问题的多种表达方式，从而提高检索系统的召回率（Recall）。我们可以通过从不同角度重写问题，可以优化查询的表述方式，使其更符合文档的索引方式。在 ​​RAG-Fusion​​ 中，多个重写后的查询会被并行检索，最终根据倒数秩排序融合结果；在 ​​Multi-Query​​ 中，多个等价查询会被（串行或并行）检索，结果也会被去重合并。前者注重于提高检索的效率和效果，适合于需要针对复杂问题提供精确答案的场景，后者的目标是提高检索的覆盖率，适合需要覆盖广泛信息的场景，尤其是当用户问题涉及多个方面的知识时。**

- Sub-question：将问题分解为更低层级的子问题（Least-to-Most）。

    **用户的问题可能过于复杂或涉及多个方面，直接检索可能导致结果不够精确。通过将问题分解为多个子问题，可以逐步缩小检索范围，找到更具体的答案。常见的子问题分解方法包括 ​​Least-to-Most​​，即从最宽泛的问题开始，逐步细化到最具体的问题。**
    
    **LangChain中的示例的子问题分解方案采用了递进填充式的Prompt，遍历问题列表，每一趟的Prompt由 当前遍历到的子问题 + 历史QA对 + 当前问题的retrieval背景知识 组成；另一种简单的方法则是直接独立运行三个subquery，简单拼接形成带有子问题QA对的Prompt，这种方法没有递进。**

## ​​Pseudo-documents
​Pseudo-documents（伪文档）​​ 是一种在信息检索、自然语言处理（NLP）和推荐系统中常用的概念。在这种方式中，可以通过某种方式生成模拟真实文档的虚拟文档来提高检索效果。
- HyDE：生成一个 ​假设文档​​来增强检索效果，从而解决原始问题在检索过程中可能面临的稀疏性、语义表达不足等问题。

    **非常直观的想法，从问题生成目标假设文本，用这个文本进行retrieval，这可以提高检索的效果，然后将检索结果融入context再提出原始问题。**

# Routing
将Query Translation得到的问题集合导航到合适的数据源。但是宏观上，路由不一定是routing到对应的数据源，你可以将query转发到任何地方（比如一个prompt，不同的个prompt下是不一样的各个数据源的处理链 ），只要这样的pipeline有助于解决你的实际问题。

## Logical Routing
- Basically 将我们维护的数据源信息简单的提供给LLM，让LLM自己推理将这个问题应用到哪一个数据源进行retrieval。

    **让大模型理解问题，生成结构化输出，用结构化输出构建不同的分支导向不同的处理链。**

## Semantic Routing
- 一种简单的方法是为不同领域的问题设计不同的Prompt，然后将Prompt向量化，与问题计算相似度，选取最相似的Prompt根据问题构建查询。

## More
- 事实上这两种可以混合使用，在规则明确时依赖逻辑路由，在语义复杂时依赖语义路由。

# Query Construction
将自然语言转化为某个特定领域的特定源的处理语言（Text2SQL、Text2Cypher等等）。
- 其实就是让LLM根据Query的自然语言，提取其中用于Filter的关键信息字段，并使用结构化输出来让这些信息可以轻松的引用和调用，再构建实际查询操作。

# Indexing
## Chunk Optimization
- Fixed Size会造成不合理的截断， overlap可以缓解这个问题，Recursive Character会根据符号自动切分。
- 文档拆分器
- Semantic Chunker的核心思想是通过 ​​句子嵌入（Sentence Embedding）​​ 将文本转换为向量，计算相邻句子的语义相似度，若相似度低于动态阈值（如分位数或标准差方法），则判定为分块边界，从而生成语义连贯、信息完整的文本块，避免传统固定分块导致的语义断裂问题
- Agentic Chunking：将文本拆解为独立命题（Proposition），再由大语言模型（LLM）主动评估并动态分配命题到文本块。
## Multi Representation
- 这是一种很直观的方法，用LLM对doc进行summary的生成，再对summary部分进行embedding，执行retrieval时仅对summary的向量与问题进行匹配，将向量化的summary和字符串的原始文档通过id关联起来。
- 这样可以将完整的文档作为context提供给LLM，保证完整的背景信息，减少split带来的信息损失。
## Hierarchical Indexing
以Raptor为例， 其实就是一种针对文档构建分层索引的方法，用一个递归的过程在各个层级上 Embedding + Clustering + Summary。 
## Specialized Embeddings
与直接将文档进行向量化不同，我们将文档分为Tokens，然后对其进行向量化，同样，在处理问题时，将问题也分为不同的Tokens，然后挨个进行相似度搜索。

# Active RAG
Active RAG 强调模型在对话或任务过程中能主动“意识到”当前信息是否足够，并做出“主动检索”的决策。

**LLM decides when and what to retrieve based upon retrieval and / or generation**

简单来说，从单纯的Feed Prompt——Generation变成一个带loops的状态机，所有的状态变换都在代码中完成。

## CRAG（Corrective RAG）
Corrective RAG 是一种对经典 RAG 框架的增强，它的目标是通过后处理机制纠正初始生成中的错误或遗漏，提高生成内容的准确性和完整性。重点是reasoning about the documents。
 
# Adaptive RAG
完整的状态机。