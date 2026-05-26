package ht.mbds.tp4testcharles.llm;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;


import java.io.Serializable;

public class LlmClientPourGemini implements Serializable {

//private final GuideTouristique guide;

    public LlmClientPourGemini() {
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        GoogleAiGeminiChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_API_KEY"))
                .modelName("gemini-2.5-flash")
                .responseFormat(ResponseFormat.JSON)
                .build();
    }

    public String chat(String question, int nb) {
        return "";
    }
}
