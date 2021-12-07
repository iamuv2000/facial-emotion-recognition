from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model 
import plotly.graph_objects as go 
import plotly.offline as plo
from plotly import subplots
from skimage import io
import tkinter
import random

model = load_model('model.h5')

def emotion_analysis(emotions): 
    objects = ['Neutral', 'Disgust', 'Fear', 'Sad', 'Happy', 'Surprise', 'Angry']
    y_pos = np.arange(len(objects))

    fig1 = go.Bar(x=objects, y=emotions*100, marker={'color': 'crimson'}, showlegend=False, name="")

    fig2 = go.Pie(labels=objects, values=emotions*100, name="")

    fig3 = go.Funnel(y=objects,x=emotions*100,name="",marker={'color' : 'tan'}, showlegend=False) #funnel

    fig4 = go.Scatter(x=objects, y=emotions*100, name="",marker={'color' : 'teal'}, showlegend=False, fill= 'tonexty', fillcolor='rgb(111, 231, 219)')

    fig5 = go.Scatter(x=objects, y=emotions*100, name="",mode= 'markers', marker={'color' : 'yellow', 'size' : emotions*150}, showlegend=False) 

    img = io.imread("happy.jpg")
    fig6 = go.Image(z=img)
    figure = subplots.make_subplots(
    rows=2,
    cols=3,
    specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "pie"}],[{"type": "scatter"}, {"type": "funnel"}, {"type": "image"}] ], subplot_titles= ("Bar Graph","Bubble Chart","Pie Chart","Area Chart","Funnel Chart","Image for Emotion Recognition") )

    figure.add_trace(fig1, 1, 1) 
    figure.add_trace(fig5, 1, 2) 
    figure.add_trace(fig2, 1, 3) 
    figure.add_trace(fig4, 2, 1)
    figure.add_trace(fig3, 2, 2) 
    figure.add_trace(fig6, 2, 3) 
    figure.update_layout(
        {
            "autosize" : True,
            "title" :{"text" : "Emotion Recognition Analysis", "font" : {"size" : 30}},
            "xaxis_title" : {"text" : "Emotion", "font" : {"size" : 20}}, 
            "yaxis_title" : {"text" : "Percentage", "font" : {"size" : 20}},
            "template" : "plotly_dark"
            }
    )
    figure.show()

file = 'happy.jpg'
true_image = image.load_img(file)
img = image.load_img(file, color_mode = "grayscale", target_size=(48, 48))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

happy_message = {
1 : "Hey! Keep smiling.",
2 : "A smile is curve that can set everything straight! Keep smiling ",
3 : "Keep smiling! It’s free therapy !",
4 : "You’ll find that life is still worthwile, if you just smiling!",
6 : "A smile is happiness that you’ll find right under your nose! Keep smiling",
7 : "Hoping that you always find a reason to smile ",
8 : "Life is a camera, so keep smiling!",
9 : "Always smile, because your smile is reason for others to smile! ", 10 : "Life is short! Smile while you still have teeth ",
}

angry_message = {
1 : "Now the sky is overcast, but soon the sun will appear. The problems do not last forever, everything will be fine!",
2 : "Every time you feel upset, remember that God does not give us trials that we cannot withstand.",
3 : "“Don’t give up when dark times come. The more storms you face in life, the stronger you’ll be. Hold on. Your greater is coming.”",
4 : "Cheer up, my dear. After every storm comes the sun. Happiness is waiting for you ahead.",
5 : "If it comes, let it come. If it goes, it's ok, let it go. Let things com e and go. Stay calm, don't let anything disturb your peace, and carry on",
6 : "Getting angry in a stressful situation is like trying to clean something with dirt",
7 : "Free your hearts of anxiety, pain and anger, to have peace within your h eart and soul",
8 : "Reacting in anger or annoyance will not advance one's ability to persuad e.",
9 : "When one burns one's bridges, what a very nice fire it makes",
10 : "Calm mind brings inner strength and self- confidence, so that’s very important for good health.", 
}

fear_message = {
1 : "Courage is knowing what not to fear.",
2 : "“Fears are educated into us, and can, if we wish, be educated out.”",
3 : "“Curiosity will conquer fear even more than bravery will.”",
4 : "“Fear: False Evidence Appearing Real.”",
5 : "“I am not afraid of tomorrow, for I have seen yesterday and I love today.”— William Allen White",
6 : "“Laughter is poison to fear, so cheer up my friend”",
7 : "“Fear is only as deep as the mind allows.”",
8 : "“One of the greatest discoveries a man makes, one of his great surprises , is to find he can do what he was afraid he couldn’t do, so don’t stop trying”",
9 : "The only thing we have to fear is fear itself",
10 : "“To overcome fear, here’s all , and do the action you fear anyway.”", 
}

surprise_message = {
1 : "Oh you seem surprised! Hope it was something good :)",
2 : "HEY! Hope you liked the surprise!",
3 : "Oh definitely share that happiness you got when you were surprised!", 4 : "Hey! Please do share the story of your surprised reaction!",
5 : "Haha who knew! Hope you like that surprise!",
6 : "Your expression tells that you were surprised by something! Hope it was good",
7 : "Gotcha! ",
8 : "Unbelievable! You never saw it coming did you?!", 
9 : "That caught you off guard! ",
10 : "You seem to be rooted to the spot! ",
}

neutral_message = {
1 : "Here let me tell you a joke: What shoes do bears wear? They don't, they go bear feet.",
2 : "What kind of clothes do houses wear? Adress.",
3 : "Hey let’s put a smile on that face! ",
4 : "Here’s a joke: I forgot how to throw a boomerang, but it came back to me.",
5 : "There’s always a reason to smile! Find it 😀",
6 : "Why don't skeletons watch scary movies? Because they don't have the guts",
7 : "What you call an owl that does magic? HOO-Dini ",
8 : "How do you call a group of unorganized cats? A cat-astrophe.", 
9 : "Life is short! Smile while you still have teeth",
10 : "What's barber's favorite instrument? A hair-monica.",
}

sad_message = {
1 : "And just like any other hard time, you’ll make it through this one too. Life is tough but so are you!",
2 : "Life is tough but so are you!",
3 : "Everything is going to work out just fine",
4 : "Sad Message 4",
5 : "Hey life is a gift, don’t waste it in melancholy",
6 : "Just fight a little longer my friend, it’s all worth it in the end. ", 7 : "Be proud of how hard you are trying",
8 : "Hang in there, it’s astonishing how short a time wonderful things take to happen.",
9 : "What did tomato say to the other tomato during a race? Ketchup.", 
10 : "Why did the bicycle fall over? Because it was two tired.",
}

disgust_message = {
1 : "Hey please don’t waste your time on being disgusted by something as pety!",
2 : "Relish everything that's inside of you, the imperfections, the darkness, the richness and light and everything. And that makes for a full life.",
3 : "Live daringly, boldly, fearlessly. Taste the relish to be found in competition - in having put forth the best within you.",
4 : "Look forward to your day! Let bygones be bygones",
5 : "Discuss the situation with someone and you’ll find yourself smiling in no time.",
6 : "Take control of that disgust! Don't let it lead your day or life.",
7 : "Try not to be disgusted with people, it can lead to grave social consequences.",
8 : "Don’t let the disgust engulf your thinking! Move on!",
9 : "Hey be optimistic and be productive.",
10 : "Expose yourself to that object of aversion, to the point where it doesn’t bother you. Face it confidently! ", 
}

custom = model.predict(x) 
expression = list(custom[0]*100) 
emotion_analysis(custom[0])

root = tkinter.Tk() 
root.title("Message")
root.geometry("300x200")
var = tkinter.StringVar()
label = tkinter.Message( root, textvariable=var, relief= tkinter.RAISED, width= 300)

# ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
if expression.index(max(expression)) == 0:
    var.set("Neutral : " + neutral_message[random.randint(1, 10)])
elif expression.index(max(expression)) == 1:
    var.set("Disgust : " + disgust_message[random.randint(1, 10)])
elif expression.index(max(expression)) == 2:
    var.set("Fear : " + fear_message[random.randint(1, 10)]) 
elif expression.index(max(expression)) == 3:
    var.set("Sad : " + sad_message[random.randint(1, 10)]) 
elif expression.index(max(expression)) == 4:
    var.set("Happy : " + happy_message[random.randint(1, 10)]) 
elif expression.index(max(expression)) == 5:
    var.set("Surprise : " + surprise_message[random.randint(1, 10)]) 
elif expression.index(max(expression)) == 6:
    var.set("Angry : " + angry_message[random.randint(1, 10)])

label.pack()
root.mainloop()