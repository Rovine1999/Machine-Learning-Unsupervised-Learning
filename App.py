from unittest import result
import joblib ### load models into the app
import streamlit as st

classifier = joblib.load('svc.pkl')

def prediction_generator(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if prediction == 0:
        output = 'Iris-setosa'

    elif prediction == 1:
        output = 'Iris-versicolor'
    
    else:
        output = 'Iris-virginica'

    return output

def main():
    st.title('Plant Species Detection App')
    sepal_length = float(st.number_input('Sepal Length(Cm)'))
    sepal_width = float(st.number_input('Sepal Width(Cm)'))
    petal_length = float(st.number_input('Petal Length(Cm)'))
    petal_width = float(st.number_input('Petal Width(Cm)'))

    result = ''

    if st.button('predict'):
        result = prediction_generator(sepal_length, sepal_width, petal_length, petal_width)
        st.success(f'The Plant is an {result}')


if __name__ == '__main__':
        main()
