
> ## Documentation Index
> Fetch the complete documentation index at: https://docs.pinecone.io/llms.txt
> Use this file to discover all available pages before exploring further.

# Concepts

>理解Pinecone中的各种概念以及之间的关系
> 
> Understand concepts in Pinecone and how they relate to each other.

<img className="block fix dark:hidden"  src="https://mintcdn.com/pinecone/r0TaYXrfSrAYZYUj/images/objects.png?fit=max&auto=format&n=r0TaYXrfSrAYZYUj&q=85&s=6a1fed2139efe425406492dd141479e5" width="1560" height="1080" data-path="images/objects.png" />

<img className="hidden max-w-full dark:block" noZoom src="https://mintcdn.com/pinecone/r0TaYXrfSrAYZYUj/images/objects-dark.png?fit=max&auto=format&n=r0TaYXrfSrAYZYUj&q=85&s=361fa6648a258883c2265e90a22465bb" width="1560" height="1080" data-path="images/objects-dark.png" />



## Project

一个项目属于一个组织, 包含多个索引, 只有项目的用户才能获取索引  API key 和Assistant都是专属的



A project belongs to an [organization](#organization) and contains one or more [indexes](#index). Each project belongs to exactly one organization, but only [users](#user) who belong to the project can access the indexes in that project. [API keys](#api-key) and [Assistants](#assistant) are project-specific.

For more information, see [Understanding projects](/guides/projects/understanding-projects).

## Index

索引有两种: 密集型和稀疏型

There are two types of [serverless indexes](/guides/index-data/indexing-overview), dense and sparse.

### Dense index

密集索引都保存了一个密集向量

Dense indexes store records that have one [dense vector](#dense-vector) each.

<Note>

同时拥有稀疏型和密集型的索引称为混合型If any records in a dense index also have a [sparse vector](#sparse-vector), the index is a [hybrid index](/guides/search/hybrid-search#use-a-single-hybrid-index).
</Note>


密集向量是一系列数字，表示文本、图像或其他类型数据的含义和关系。密集向量中的每个数字对应于多维空间中的一个点。在该空间中距离较近的向量在语义上是相似的。

当您查询密集索引时，Pinecone 会检索包含在语义上与查询最相似的密集向量的记录。这通常称为语义搜索、最近邻搜索、相似性搜索或向量搜索。
A dense vector is a series of numbers that represent the meaning and relationships of text, images, or other types of data. Each number in a dense vector corresponds to a point in a multidimensional space. Vectors that are closer together in that space are semantically similar.

当您查询密集索引时，Pinecone 会检索包含在语义上与查询最相似的密集向量的记录。这通常称为语义搜索、最近邻搜索、相似性搜索或向量搜索
When you query a dense index, Pinecone retrieves records containing dense vectors that are most semantically similar to the query. This is often called **semantic search**, nearest neighbor search, similarity search, or just vector search.

### Sparse index

Sparse indexes store records that have one [sparse vector](#sparse-vector) each.

每个稀疏型向量都是一系列数字, 用以代表文档中的字句或者段落

稀疏向量具有非常大的维度，其中只有一小部分值是非零的。

维度代表字典中的单词，值代表这些单词在文档中的重要性。

Every sparse vector is a series of numbers that represent the words or phrases in a document. Sparse vectors have a very large number of dimensions, where only a small proportion of values are non-zero. The dimensions represent words from a dictionary, and the values represent the importance of these words in the document.

当您搜索稀疏索引时，Pinecone 会检索具有与查询中的单词或短语最匹配的稀疏向量的记录。查询术语独立评分，然后求和，具有最相似向量的记录得分最高。这通常称为**词法搜索**或**关键字搜索**

When you search a sparse index, Pinecone retrieves the records with sparse vectors that most exactly match the words or phrases in the query. Query terms are scored independently and then summed, with records that have the most similar vectors scored highest. This is often called **lexical search** or **keyword search**.

## Namespace



命名空间密集索引或稀疏索引中的分区。它将索引中的记录分为不同的组

所有的上传 查询 和其他数据操作始终先确定命名空间

A namespace is a partition within a [dense](#dense-index) or [sparse index](#sparse-index). It divides [records](#record) in an index into separate groups.

All [upserts](/guides/index-data/upsert-data), [queries](/guides/search/search-overview), and other [data operations](/reference/api/latest/data-plane) always target one namespace:

<img className="block max-w-full dark:hidden" noZoom src="https://mintcdn.com/pinecone/r0TaYXrfSrAYZYUj/images/quickstart-upsert.png?fit=max&auto=format&n=r0TaYXrfSrAYZYUj&q=85&s=641c2aa9a3238bf70698c583097c1f29" width="1400" height="880" data-path="images/quickstart-upsert.png" />

<img className="hidden max-w-full dark:block" noZoom src="https://mintcdn.com/pinecone/r0TaYXrfSrAYZYUj/images/quickstart-upsert-dark.png?fit=max&auto=format&n=r0TaYXrfSrAYZYUj&q=85&s=14a3e6c2847455db0821ebbf9bd51df9" width="1400" height="880" data-path="images/quickstart-upsert-dark.png" />

For more information, see [Use namespaces](/guides/index-data/indexing-overview#namespaces).

## Record

A record is a basic unit of data and consists of a [record ID](#record-id), a [dense vector](#dense-vector) or a [sparse vector](#sparse-vector) (depending on the type of index), and optional [metadata](#metadata).

基础数据单元, 密集和稀疏向量和其他元数据都有记录

For more information, see [Upsert data](/guides/index-data/upsert-data).

### Record ID

A record ID is a record's unique ID. [Use ID prefixes](/guides/index-data/data-modeling#use-structured-ids) that reflect the type of data you're storing.

### Dense vector

密集向量

* 一大串数字, 表示数据之间的关系
* 多维度
* 在语义相似度空间中, 集合更加紧密

A dense vector, also referred to as a vector embedding or simply a vector, is a series of numbers that represent the meaning and relationships of data. Each number in a dense vector corresponds to a point in a multidimensional space. Vectors that are closer together in that space are semantically similar.

Dense vectors are stored in [dense indexes](#dense-index). 密集向量保存在密集索引中

使用密集嵌入模型把数据传入密集向量中  嵌入模型可以在Pinecone外部或者在Pinecone基础设施中通过索引集成

You use a dense embedding model to convert data to dense vectors. The embedding model can be external to Pinecone or [hosted on Pinecone infrastructure](/guides/index-data/create-an-index#embedding-models) and integrated with an index.

For more information about dense vectors, see [What are vector embeddings?](https://www.pinecone.io/learn/vector-embeddings/).

### Sparse vector
稀疏向量常用于以能够捕获关键词信息的方式表示文档或查询。

稀疏向量中的每个维度通常代表词典中的一个单词，而非零值则代表这些单词在文档中的重要性。稀疏向量具有大量维度，但其中只有少数值是非零的。

由于大多数值为零，Pinecone 通过仅保留非零值及其对应的索引来高效存储稀疏向量。

稀疏向量存储在稀疏索引和混合索引中。要将数据转换为稀疏向量，请使用稀疏嵌入模型。该嵌入模型可以是 Pinecone 外部的，也可以托管在 Pinecone 基础设施上并与索引集成。


Sparse vectors are often used to represent documents or queries in a way that captures keyword information. Each dimension in a sparse vector typically represents a word from a dictionary, and the non-zero values represent the importance of these words in the document.

Sparse vectors have a large number of dimensions, but a small number of those values are non-zero. Because most values are zero, Pinecone stores sparse vectors efficiently by keeping only the non-zero values along with their corresponding indices.

Sparse vectors are stored in [sparse indexes](#sparse-index) and [hybrid indexes](/guides/search/hybrid-search#use-a-single-hybrid-index). To convert data to sparse vectors, use a sparse embedding model. The embedding model can be external to Pinecone or [hosted on Pinecone infrastructure](/guides/index-data/create-an-index#embedding-models) and integrated with an index.

For more information about sparse vectors, see [Sparse retrieval](https://www.pinecone.io/learn/sparse-retrieval/).

### Metadata

元数据是额外信息 包含了提供更多内容的记录 和 额外的过滤能力 比如嵌入的原文本可以通过元数据分类

Metadata is additional information included in a record to provide more context and enable additional [filtering capabilities](/guides/index-data/indexing-overview#metadata). For example, the original text that was embedded can be stored in the metadata.

## Other concepts

Although not represented in the diagram above, Pinecone also contains the following concepts:

* [API key](#api-key)
* [User](#user)
* [Backup or collection](#backup-or-collection)
* [Pinecone Inference](#pinecone-inference)

### API key

An API key is a unique token that [authenticates](/reference/api/authentication) and authorizes access to the [Pinecone APIs](/reference/api/introduction). API keys are project-specific.

### User

A user is a member of organizations and projects. Users are assigned specific roles at the organization and project levels that determine the user's permissions in the [Pinecone console](https://app.pinecone.io).

For more information, see [Manage organization members](/guides/organizations/manage-organization-members) and [Manage project members](/guides/projects/manage-project-members).

### Backup or collection



备份是无服务器索引的静态副本。



备份仅占用存储空间。它们是一组记录的不可查询表示形式。您可以从索引创建备份，也可以基于该备份创建新的索引。新索引的配置可以与原始源索引不同：例如，它可以拥有不同的名称。但是，它必须与源索引具有相同数量的维度和相似度度量。

A backup is a static copy of a serverless index.

Backups only consume storage. They are non-queryable representations of a set of records. You can create a backup from an index, and you can create a new index from that backup. The new index configuration can differ from the original source index: for example, it can have a different name. However, it must have the same number of dimensions and similarity metric as the source index.

For more information, see [Understanding backups](/guides/manage-data/backups-overview).

### Pinecone Inference

Pinecone Inference is an API service that provides access to [embedding models](/guides/index-data/create-an-index#embedding-models) and [reranking models](/guides/search/rerank-results#reranking-models) hosted on Pinecone's infrastructure.

## Learn more

* [Vector database](https://www.pinecone.io/learn/vector-database/)
* [Pinecone APIs](/reference/api/introduction)
* [Approximate nearest neighbor (ANN) algorithms](https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/)
* [Retrieval augmented generation (RAG)](https://www.pinecone.io/learn/retrieval-augmented-generation/)
* [Image search](https://www.pinecone.io/learn/series/image-search/)
* [Tokenization](https://www.pinecone.io/learn/tokenization/)
