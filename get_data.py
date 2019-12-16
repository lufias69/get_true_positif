from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

def get_data(a, b, data, class_ = "spam"):
    data = np.array(data)
    same_list = list()
    index = 0
    for i, j in zip(a,b):
        if i==j and i == class_:
            same_list.append(index)
        index+=1
    data = data[same_list]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    fitur = vectorizer.get_feature_names()
    X = X.sum(0)
    X= {"fitur":fitur, "jumlah":X.tolist()[0]}
    return pd.DataFrame.from_dict(X)
