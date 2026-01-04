import { GoogleGenAI } from "@google/genai";

const apiKey = "AIzaSyBGdpw3KuRFiLzxNIPjCMphUzlInLLx-Pk"; 
const client = new GoogleGenAI({ apiKey });

console.log("Gemini Client Initialized with @google/genai");

// Default model if not specified
let activeModel = "gemini-3-flash-preview";

export const updateModel = (model: string) => {
    activeModel = model;
    console.log(`Switched Gemini Model to: ${activeModel}`);
};

export const getModelName = () => activeModel;

// Helper function to safely get text from response
const getTextFromResponse = (response: any) => {
    if (!response) return "No response";
    if (typeof response.text === 'function') return response.text();
    if (response.text) return response.text;
    // Fallback for different SDK response structures
    if (response.candidates && response.candidates[0]?.content?.parts?.[0]?.text) {
        return response.candidates[0].content.parts[0].text;
    }
    return JSON.stringify(response);
};

export const generateVoiceSimResponse = async (
    history: { role: "user" | "model", content: string }[],
    lastUserAudioInput: string | null = null
) => {
    // Note: For this version, we simulate the interaction using text prompts if audio model isn't fully supported via REST/SDK yet 
    // or if we treat 'lastUserAudioInput' as transcribed text.
    // Ideally with "gemini-2.5-flash-native-audio-preview-09-2025" we would send audio blob directly.
    // For now, we assume STT happens on client (Web Speech API) and we send text.
    
    // If the user explicitly selected the audio model, we try to use it, 
    // otherwise fallback to activeModel or a text optimized one.
    
    const modelToUse = activeModel.includes("audio") ? activeModel : "gemini-2.0-flash"; 

    const systemInstruction = `
        You are an AI Interviewer conducting a job interview for a Product Manager position in an AI company.
        Your candidate has B1 English level, so speak clearly but professionally.
        
        Role: Professional, encouraging but rigorous Hiring Manager.
        Topic: AI Product Management, LangChain, RAG, Voice AI, Agency experience.
        
        Goal: 
        1. Ask one relevant question based on the context.
        2. Wait for user answer.
        3. Provide brief constructive feedback on their answer (grammar/content).
        4. Ask the next question.
        
        Keep your responses short (under 3 sentences) so it feels like a real conversation.
    `;

    const chatHistory = history.map(h => ({
        role: h.role === "user" ? "user" : "model",
        parts: [{ text: h.content }]
    }));

    try {
        console.log(`Generating Voice Sim Response using model: ${modelToUse}`);
        // Using standard generateContent for chat-like behavior stateless or chatSession
        // Here we construct a simple prompt chain for stateless request or use chat if supported
        
        // Simulating chat via single prompt for simplicity/robustness in this snippet
        const prompt = `
            ${systemInstruction}
            
            Conversation History:
            ${history.map(h => `${h.role.toUpperCase()}: ${h.content}`).join("\n")}
            
            USER (Latest): ${lastUserAudioInput || "(Waiting for question)"}
            
            AI INTERVIEWER RESPONSE:
        `;

        const response = await client.models.generateContent({
            model: modelToUse,
            contents: prompt,
        });
        
        return getTextFromResponse(response);

    } catch (error) {
        console.error("Gemini Voice Sim Error:", error);
        return "Error connecting to AI Interviewer. Please try again.";
    }
};

export const generateTailoredResume = async (
    baseResume: string,
    companyName: string,
    companyData: any,
    referenceResume: string | null = null
) => {
    const prompt = `
      You are an expert Resume Writer and Career Coach.
      
      Goal: Rewrite the candidate's "Base Resume" to perfectly target the "Target Company".
      
      Context:
      - Target Company: ${companyName}
      - Company Data (Pain Points, Needs): ${JSON.stringify(companyData.strategy_analysis || {})}
      - Company Roles: ${JSON.stringify(companyData.open_roles || [])}
      
      ${referenceResume ? `
      - STYLE REFERENCE: Please Adopt the structure, formatting, and tone of the "Reference Resume" provided below, but strictly using the candidate's facts from "Base Resume".
      - Reference Resume Content:
      """${referenceResume}"""
      ` : `
      - Style: Professional, impact-oriented, highlighting "Vibecoding" and "Agency Owner" strengths as assets for a Product Manager role.
      `}
      
      - Base Resume Content:
      """${baseResume}"""
      
      Instructions:
      1. Keep the candidate's core facts (don't invent jobs).
      2. Rephrase bullet points to match the company's "Pain Points" and "Keywords".
      3. If a Reference Resume is provided, COPY its layout (headers, order) and stylistic choices (e.g. short bullets vs long paragraphs) exactly.
      4. Highlight specific skills that match the company (e.g. if they need LangChain, emphasize it).
      
      Output:
      A full, ready-to-copy Resume in Markdown format.
    `;

    try {
        console.log(`Generating Tailored Resume for ${companyName} using model: ${activeModel}`);
        const response = await client.models.generateContent({
            model: activeModel,
            contents: prompt,
        });
        return getTextFromResponse(response);
    } catch (error) {
        console.error("Gemini Resume Gen Error:", error);
        return "Error generating resume.";
    }
};

export const generateCoverLetter = async (
  companyName: string,
  role: string,
  painPoints: string[],
  userStrengths: string[],
  killerFeature: string,
  tone: "casual" | "professional" | "confident" = "professional"
) => {
  const prompt = `
    You are an expert career coach for a Product Manager who is a "Vibe Coder" (practical technical skills, no CS degree) and has B1 English level.
    
    Your goal is to write a "Simple & Punchy" cover letter / cold message.
    It must be short, use active verbs, and avoid complex grammar (B1 friendly).
    
    Context:
    - Target Company: ${companyName}
    - Role: ${role}
    - Their Likely Pain Points: ${painPoints.join(", ")}
    - My Key Strengths: ${userStrengths.join(", ")}
    - My "Secret Weapon" / Killer Feature idea: ${killerFeature}
    
    Tone: ${tone} (But always direct and result-oriented).
    
    Structure:
    1. Hook: Mention a specific problem or observation about their product/market.
    2. Value: How I solved a similar problem or my specific idea for them.
    3. Proof: One sentence with a metric or "Vibe Coding" achievement.
    4. Call to Action: "Can I show you a demo?" or "Open to a 15-min chat?"

    Output only the email body text. Keep it under 150 words.
  `;

  try {
    console.log(`Generating Cover Letter using model: ${activeModel}`);
    const response = await client.models.generateContent({
      model: activeModel, 
      contents: prompt,
    });
    return getTextFromResponse(response);
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Error generating text. Please check API key or model availability.";
  }
};

export const generateTrojanHorseStrategy = async (
  companyName: string,
  killerFeature: string,
  painPoints: string[]
) => {
  const prompt = `
    You are a Senior AI Architect and Product Manager.
    
    Your task: Create a "Trojan Horse" Technical Proposal (PRD & Architecture) for a feature that solves a critical pain point for ${companyName}.
    This proposal is meant to be sent by a candidate to impress the hiring manager.

    
    Context:
    - Company: ${companyName}
    - Feature Idea: ${killerFeature}
    - Pain Points Solved: ${painPoints.join(", ")}

    Output Format:
    Generate a structured text document using Markdown.

    Sections:
    1. **🚀 The "Trojan Horse" Concept**: One sentence pitch.
    2. **⚠️ Problem Analysis**: Why this is hard/important (technical depth).
    3. **🛠️ Proposed Solution (The "Vibe Code")**:
       - Architecture Diagram (use Mermaid syntax or clear text description of stack).
       - Key Components (e.g., "LangChain Router", "Pinecone Index", "FastAPI wrapper").
    4. **🧪 Implementation Plan (MVP)**:
       - Step 1: Data Ingestion...
       - Step 2: RAG Pipeline...
       - Step 3: UI/Demo...
    5. **🔮 Expected Impact**: Metrics (e.g., "Reduce hallucination by 30%", "Latency < 200ms").

    Tone: Highly technical but concise. Show, don't just tell.
  `;

  try {
    console.log(`Generating Strategy using model: ${activeModel}`);
    const response = await client.models.generateContent({
      model: activeModel,
      contents: prompt,
    });
    return getTextFromResponse(response);
  } catch (error) {
    console.error("Gemini API Error (Trojan Horse):", error);
    return "Error generating strategy. Please check API connection.";
  }
};

export const generatePersonaIntro = async (
    idealPersona: any,
    userBackground: any
) => {
    const prompt = `
      You are a Communication Coach helping a Product Manager candidate (B1 English level) prepare for an interview.
      
      Goal: Generate a "Tell me about yourself" intro script that perfectly bridges the candidate's background with the ideal persona the companies are looking for.
      
      Input Data:
      - Ideal Persona Pattern (What they want): ${JSON.stringify(idealPersona)}
      - My Background (Who I am): ${JSON.stringify(userBackground)}
      
      Constraint:
      - Language: Simple, clear English (B1 level). No complex idioms.
      - Tone: Confident, practical ("Vibecoder" style).
      - Structure: Past (Agency/SEO) -> Pivot (Why AI) -> Future (Why I fit this Ideal Persona).
      
      Output:
      A 2-minute spoken intro script.
    `;

    try {
        console.log(`Generating Persona Intro using model: ${activeModel}`);
        const response = await client.models.generateContent({
          model: activeModel,
          contents: prompt,
        });
        return getTextFromResponse(response);
      } catch (error) {
        console.error("Gemini API Error:", error);
        return "Error generating intro.";
      }
}

export const generateInterviewPhrases = async (
    category: string
) => {
    const prompt = `
      Generate 5 key English interview phrases/sentences for a Product Manager with B1 English level.
      Category: ${category} (e.g., "Discussing Technical Architecture", "Handling Disagreement", "Describing Metrics").
      
      Format:
      - Phrase (Simple English)
      - Why it works (Short explanation)
      
      Example:
      "I prioritized X because of Y." -> Shows decision making.
    `;

    try {
        console.log(`Generating Phrases using model: ${activeModel}`);
        const response = await client.models.generateContent({
          model: activeModel,
          contents: prompt,
        });
        return getTextFromResponse(response);
      } catch (error) {
        console.error("Gemini API Error:", error);
        return "Error generating phrases.";
      }
}

export const generatePersonaAnalysis = async (
    companyName: string,
    role: string
) => {
    const prompt = `
      Analyze the corporate culture and hiring patterns for ${companyName} (specifically for ${role} roles).
      
      Target Audience: A "Vibe Coder" PM (non-CS technical PM) with B1 English who wants to mirror their ideal employee persona.

      Output Format: Markdown.

      Sections:
      1. **🧬 The Archetype**: Describe the typical successful PM there in 2 words (e.g., "The Academic Engineer", "The Hustler", "The Data Nerd"). Explain the "Vibe".
      2. **🔑 Keywords & Concepts**: 5-7 words they use internally or in job descriptions (e.g., "First Principles", "Bias for Action", "Latency", "Tokens").
      3. **🗣️ English Power Phrases (B1 Safe)**: 
         - List 5 phrases to use in an interview to sound like a native pro. 
         - Keep grammar simple but vocabulary professional.
         - Example: Instead of "I managed the team", suggest "I drove alignment".
      4. **🚩 Red Flags to Avoid**: What turns them off? (e.g., "Talking too much about process", "Not knowing the API").
      
      Focus on practical, actionable advice for an interview.
    `;

    try {
        console.log(`Generating Analysis using model: ${activeModel}`);
        const response = await client.models.generateContent({
          model: activeModel,
          contents: prompt,
        });
        
        return getTextFromResponse(response);
      } catch (error) {
        console.error("Gemini API Error (Persona):", error);
        return "Error generating persona analysis.";
      }
}

export const generateMarketAnalysis = async (jobs: any[]) => {
    // Prepare a simplified list of jobs to save context window
    const simplifiedJobs = jobs.map(j => ({
        title: j.title,
        requirements: j.requirements
    }));

    const prompt = `
      You are a Market Research Analyst for AI Product Management roles.
      
      Task: Analyze the provided list of job vacancies and their requirements to determine the "Average Fit" skill profile.
      
      Input Data (Vacancies):
      ${JSON.stringify(simplifiedJobs)}
      
      Goal:
      1. Identify the top 6 most critical skills/tools mentioned across these jobs.
      2. Calculate a relative "Demand Score" (0-100) for each (100 = mentioned in almost every job).
      3. Provide a strategic summary for a candidate.

      Output Format: STRICT JSON ONLY.
      Structure:
      {
        "chartData": [
          { "subject": "Skill Name", "A": number (Demand Score), "fullMark": 100 },
          ... (6 items)
        ],
        "analysis": "Markdown string with strategic advice. Keep it under 200 words. Use emoji bullet points."
      }
    `;

    try {
        console.log(`Generating Market Analysis using model: ${activeModel}`);
        const response = await client.models.generateContent({
            model: activeModel,
            contents: prompt,
        });
        const text = getTextFromResponse(response);
        
        // Extract JSON from potential markdown wrapping
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }
        return JSON.parse(text);
    } catch (error) {
        console.error("Gemini Market Analysis Error:", error);
        // Return mock data on error to prevent UI crash
        return {
            chartData: [
                { subject: 'Python', A: 90, fullMark: 100 },
                { subject: 'Product Mgmt', A: 95, fullMark: 100 },
                { subject: 'AI/ML', A: 85, fullMark: 100 },
                { subject: 'English', A: 70, fullMark: 100 },
                { subject: 'Analytics', A: 60, fullMark: 100 },
                { subject: 'SEO', A: 50, fullMark: 100 },
            ],
            analysis: "⚠️ Error connecting to AI Analyst. Displaying default baseline data. Please try again."
        };
    }
};
