_file_ = "01_basic_inline_dspy_signature_example.py"

import dspy

QUESTIONS = ["What is dark matter in the universe?",
            "Why did the dinosaurs go extinct?",
            "why did the chicken cross the road?"
]

SENTIMENTS = ["Movie was great!", 
              "Movie was terrible!",
              "Movie was okay!"]

SUMMARY = """
    DSPy is a framework for algorithmically optimizing LM prompts and weights, 
    especially when LMs are used one or more times within a pipeline. To use 
    LMs to build a complex system without DSPy, you generally have 
    to: (1) break the problem down into steps, (2) prompt your 
    LM well until each step works well in isolation, (3) tweak 
    the steps to work well together, (4) generate synthetic 
    examples to tune each step, and (5) use these examples to finetune 
    smaller LMs to cut costs. Currently, this is hard and messy: every 
    time you change your pipeline, your LM, or your data, all prompts 
    (or finetuning steps) may need to change.

    To make this more systematic and much more powerful, DSPy does two things. 
    First, it separates the flow of your program (modules) from the 
    parameters (LM prompts and weights) of each step. 
    Second, DSPy introduces new optimizers, which are LM-driven 
    algorithms that can tune the prompts and/or the weights of 
    your LM calls, given a metric you want to maximize.

    DSPy can routinely teach powerful models like GPT-3.5 or GPT-4 and local 
    models like T5-base or Llama2-13b to be much more reliable at tasks, i.e. having 
    higher quality and/or avoiding specific failure patterns. DSPy optimizers will "compile" 
    the same program into different instructions, few-shot prompts, and/or weight 
    updates (finetunes) for each LM. This is a new paradigm in which LMs and their prompts 
    fade into the background as optimizable pieces of a larger system that can learn from data. 
    tldr; less prompting, higher scores, and a more systematic approach to solving hard tasks 
    with LMs.
"""

if __name__ == "__main__":

    from columbus_api import Columbus
    columbus = Columbus()
    llm = columbus.get_llm_for_DSPy("gpt-4-turbo")
    llm.kwargs['max_tokens']=1500
    
    # Setup Ollama environment
    # ollama_mistral = dspy.OllamaLocal(model='mistral')
    # dspy.settings.configure(lm=ollama_mistral)
    dspy.settings.configure(lm=llm)

    # Use inline signatgure for question answering
    for question in QUESTIONS:
        answer = dspy.Predict('question -> answer')
        print(f"Question: {question}")
        print(f"Answer: {answer(question=question).answer}")
        print("-------------------")

    # Use line signatures for classification
    for sentiment in SENTIMENTS:
        classify = dspy.Predict('sentence -> sentiment')
        print(f"{classify(sentence=sentiment).sentiment}")
        print("-------------------")

    # use line signatures for summarization
    summarize = dspy.Predict('text -> summary')
    print("Summary:")
    print(summarize(text=SUMMARY).summary)
    
    # simply joke telling
    joker = dspy.Predict('hint -> joke')
    print("Get ready to rof:")
    print(joker(hint="chair").joke)

    llm.inspect_history(n=1)

    