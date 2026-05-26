package ht.mbds.tp4testcharles.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

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

        // -------------------------------------------------------
        // PHASE 2 : utilisation des embeddings pour répondre aux questions
        // -------------------------------------------------------

        // Création du ContentRetriever : 2 résultats max, score minimum 0.5
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Création de la mémoire pour 10 messages
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Création de l'assistant avec le ContentRetriever
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(chatMemory)
                .contentRetriever(retriever)
                .build();

        conversationAvec(assistant);
    }

    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
