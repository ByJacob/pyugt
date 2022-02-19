from transformers import MarianMTModel, MarianTokenizer

def split_by_characters(text):
    result = []
    actual = ""
    for idx in range(len(text)-1):
        ch = text[idx]
        if not ch.isalnum() and ch not in " .?!'":
            continue
        next_ch = text[idx+1]
        actual += ch
        if ch in ".?!" and next_ch == " ":
            result.append(actual.strip()[0].upper()+actual.strip()[1:])
            actual = ""
    result.append(actual.strip()[0].upper() + actual.strip()[1:])
    return result


src_text = "Oops! Talking too much again.Why don't you take a look around? You might uncover something interesting!"
src_texts = split_by_characters(src_text)
model_name = "gsarti/opus-tatoeba-eng-pol"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

print("Start Trans - model generate")
translated = model.generate(**tokenizer(src_texts, return_tensors="pt", padding=True))
print("tokenizer decode")
test = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(test)