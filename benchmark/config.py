import os

MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2-2.7B": "microsoft/phi-2",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-2-13B": "meta-llama/Llama-2-13b-chat-hf",
}

# The same 20 questions used in the project
STANDARD_PROMPTS = [
    "What is photosynthesis?",
    "Explain Newton's third law of motion.",
    "How does a neural network work?",
    "What are the main causes of climate change?",
    "Describe the process of cellular respiration.",
    "What is the theory of relativity?",
    "Explain the water cycle.",
    "How do vaccines work?",
    "What is the structure of DNA?",
    "Describe the history of the Roman Empire.",
    "What are black holes?",
    "Explain the concept of supply and demand.",
    "How does a quantum computer work?",
    "What is the greenhouse effect?",
    "Describe the human digestive system.",
    "What is the significance of the Magna Carta?",
    "Explain the difference between mitosis and meiosis.",
    "How do airplanes fly?",
    "What is the periodic table?",
    "Describe the French Revolution.",
]

SHORT_PROMPTS = [
    "What is the capital of France?",
    "Name three primary colors.",
    "Who wrote Hamlet?",
    "What is H2O?",
    "How many continents are there?",
]

MEDIUM_PROMPTS = [
    "Explain the basic rules of chess and how each piece moves on the board.",
    "Describe the main differences between a democracy and a republic.",
    "What are the health benefits of regular cardiovascular exercise?",
    "How does a refrigerator keep food cold? Explain the basic thermodynamics.",
    "Summarize the plot of George Orwell's 1984 in a few sentences.",
]

LONG_PROMPTS = [
    "Please provide a comprehensive overview of the causes and consequences of the Industrial Revolution. Discuss the technological advancements, the social and economic shifts, and the long-term impact on global urbanization and living standards.",
    "Explain the principles of object-oriented programming (OOP) in detail. Discuss concepts such as encapsulation, inheritance, polymorphism, and abstraction. Provide examples of how these concepts are applied in software development.",
    "Describe the lifecycle of a star, from its formation in a nebula to its eventual demise. Explain the different stages, including main sequence, red giant, and supernova, and discuss the various end states such as white dwarfs, neutron stars, and black holes.",
    "Discuss the history and evolution of artificial intelligence. Trace its development from early symbolic AI and expert systems to the rise of machine learning and modern deep learning neural networks. Highlight key breakthroughs and ongoing challenges.",
    "Analyze the themes and symbolism in F. Scott Fitzgerald's The Great Gatsby. Focus on the representation of the American Dream, the role of wealth and class, and the significance of symbols like the green light and the Valley of Ashes.",
]

QUANTIZATION_LEVELS = ["FP16", "INT8", "INT4"]
BATCH_SIZES = [1, 2, 4]
GRID_CARBON_INTENSITY = 0.708  # kg CO2 per kWh, India average
MAX_NEW_TOKENS = 200

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CODECARBON_OUTPUT_DIR = os.path.join(DATA_DIR, "codecarbon")
