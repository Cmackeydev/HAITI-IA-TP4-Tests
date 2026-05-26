package ht.mbds.tp4testcharles.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class RagNaif {

    public static void main(String[] args) {

        String apiKey = System.getenv("GEMINI_API_KEY");

        // Création du ChatModel Gemini
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.25)
                .build();

        // -------------------------------------------------------
        // PHASE 1 : enregistrement des embeddings
        // -------------------------------------------------------

        // Chargement du PDF via Apache Tika
        ApacheTikaDocumentParser parseur = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", parseur);

        // Découpage du document en segments (chunks)
        DocumentSplitter separateur = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = separateur.split(document);

        // Création du modèle d'embedding Gemini
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-embedding-exp-03-07")
                .build();

        // Calcul des embeddings pour tous les segments
        Response<List<Embedding>> reponseEmbeddings = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = reponseEmbeddings.content();

        // Stockage des embeddings et des segments dans un magasin en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
    }
}
