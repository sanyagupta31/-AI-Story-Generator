from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_story(prompt, genre, length):
    genre_prefix = f"{genre} story: " if genre else ""
    full_prompt = genre_prefix + prompt.strip()
    
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

interface = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(label="Your story prompt", placeholder="Once upon a time..."),
        gr.Radio(
            choices=["Fantasy", "Sci-Fi", "Horror", "Romance", "Mystery", "Suspenseful", "Gory", "Psychological"],
            label="Genre",
            value="Fantasy"
        ),
        gr.Slider(minimum=50, maximum=500, step=10, value=100, label="Story Length")
    ],
    outputs="text",
    title="📝 AI Story Generator",
    description="Enter a story start and choose a genre. Let GPT-2 finish your story!",
    css=""" 
    body {
      background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
      background-size: cover;
      background-position: center;
    }
    """
)

interface.launch(share=True)
