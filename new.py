import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import emoji

from tkinter import *
from PIL import ImageTk ,Image

root = Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.config(bg="powder blue")
root.title("Text Emotion")

ent_var=StringVar()

global X_train, X_test

def read_data():
    data = []
    with open('text_emotion.txt', 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data
data = read_data()
print("Number of instances: {}".format(len(data)))

def ngram(token, n):
    output = []
    for i in range(n-1, len(token)):
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram)
    return output
def create_feature(text, nrange=(1, 1)):
    text_features = []
    text = text.lower()
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1):
        text_features += ngram(text_alphanum.split(), n)
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)


def convert_label(item, name):
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)):
        if items[idx] == 1:
            label += name[idx] + " "

    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))
# -------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 15)

def train_test(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    filename="trained_model.sav"
    pickle.dump(clf, open(filename, "wb"))

def text():
    svc = SVC()
    lsvc = LinearSVC(random_state=123)
    rforest = RandomForestClassifier(random_state=123)
    dtree = DecisionTreeClassifier()

    clifs = [svc, lsvc, rforest, dtree]
    # train and test them
    print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
    print("| {} | {} | {} |".format("-" * 25, "-" * 17, "-" * 13))

    for clf in clifs:
        clf_name = clf.__class__.__name__
        full_train=train_test(clf, X_train,  y_train)
        print(full_train)

l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
print(l)
label_freq = {}

for label, _ in data:
    label_freq[label] = label_freq.get(label, 0) + 1
#print(label_freq)

# print the labels and their counts in sorted order
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))

joy=Image.open("joy.png")
joy=joy.resize((200,200),Image.ANTIALIAS)
joy=ImageTk.PhotoImage(joy)

fear=Image.open("fear.png")
fear=fear.resize((200,200),Image.ANTIALIAS)
fear=ImageTk.PhotoImage(fear)

anger=Image.open("anger.png")
anger=anger.resize((200,200),Image.ANTIALIAS)
anger=ImageTk.PhotoImage(anger)

sadness=Image.open("sadness.png")
sadness=sadness.resize((200,200),Image.ANTIALIAS)
sadness=ImageTk.PhotoImage(sadness)

disgust=Image.open("disgust.png")
disgust=disgust.resize((200,200),Image.ANTIALIAS)
disgust=ImageTk.PhotoImage(disgust)

shame=Image.open("shame.png")
shame=shame.resize((200,200),Image.ANTIALIAS)
shame=ImageTk.PhotoImage(shame)

guilt=Image.open("guilt.png")
guilt=guilt.resize((200,200),Image.ANTIALIAS)
guilt=ImageTk.PhotoImage(guilt)

emoji_dict = {"joy":joy, "fear":fear, "anger":PhotoImage(file = r"anger.png"), "sadness":PhotoImage(file = r"sadness.png"), "disgust":PhotoImage(file = r"disgust.png"), "shame":PhotoImage(file = r"shame.png"), "guilt":PhotoImage(file = r"guilt.png")}



t1 = "This looks so impressive"
t2 = "I have a fear of dogs"
t3 = "My dog died yesterday"
t4 = "I don't love you anymore..!"

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

def abcs():

    features = create_feature(ent_var.get(), nrange=(1, 4))
    features = vectorizer.transform(features)
    model = pickle.load(open("trained_model.sav", "rb"))
    prediction = model.predict(features)[0]
    print(prediction)
    lbl2 = Label(root, image=emoji_dict[prediction])
    lbl2.place(x=(w/5+w/4), y=(h/4+h/4), height=200, width=200)






for i in ent_var.get():
    features = create_feature(i, nrange=(1, 4))
    features = vectorizer.transform(features)
    model=pickle.load(open("trained_model.sav", "rb"))
    prediction = model.predict(features)[0]
    print(prediction)
    lbl2 = Label(root, image=emoji_dict[prediction])
    lbl2.place(x=0, y=0)






lbl=Label(root, text="Dectect Text Emotion", bg="powder blue", fg="blue", font=("arial", 20, "bold")).place(x=w/4, y=h/8, height=30, width=w/2)

ent=Entry(root, fg="green", textvariable=ent_var, font=("arial", 20, "bold italic"))
ent.place(x=w/4, y=h/5, height=30, width=w/2)

btn=Button(root, text="Check Emotion",  bg="sky blue", fg="gray", font=("arial", 20, "bold italic"), command=abcs)
btn.place(x=w/3, y=h/3, height=30, width=w/3)



root.mainloop()
