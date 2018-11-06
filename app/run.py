import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as goj


app = Flask(__name__)

def tokenize(text):
    '''
    Split text to transformed tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('model', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# cheeck that the model predict
query = 'we ned food'

# use model to predict classification for query
classification_labels = model.predict([query])[0]
classification_results = dict(zip(df.columns[4:], classification_labels))
print(classification_results)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.iloc[:, 4:]
    cat_names = list(categories)
    cat_counts = [df[cat_name].sum() for cat_name in cat_names]

    categories['sum'] = categories.sum(axis=1)
    counts = categories.groupby('sum').count()['related']
    names = list(counts.index)
    
    # another graph to show the numbers in each categorie 
    categoriestwo = df.iloc[:, 4:].sum().sort_values(ascending=False)
    color_bar = 'Teal'
       
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    {
            'data': [
                Histogram(
                    y=counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Messages in several categories',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of included categories"
                },
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        },
#    ]
    
    {
        'data': [goj.Bar(
                         x=categoriestwo.index,
                         y=categoriestwo,
                        marker=dict(color='blue'),
                         opacity=0.8
                 )],
        
        'layout': goj.Layout(
                        title="Messages per Category",
                        xaxis=dict(
                        title='Categoriestwo',
                        tickangle=45
                     ),
                        yaxis=dict(
                        title='# of Messages',
                        tickfont=dict(
                        color='Black'
                            )
                            )
                            )
                            }
#            }
                           ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
    

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()