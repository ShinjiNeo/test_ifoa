import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from summarytools import dfSummary
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# per entrare in una cartella scrivere cd più percorso. es: C:\Users\ifoa\Desktop\IFOA\LIVIO_MACHINE_LEARNING\Less13
def main():
    st.title("Progetto di Data Analysis")
    st.text("Carica il dataset IRIS in formato csv o xlsx")
    file= st.file_uploader("Carica", type=['csv', 'xlsx',])
    
    if file is not None:
        if file.name.endswith('.csv'):
            df=pd.read_csv(file, header=None)
            st.success('File CSV caricato correttamente!')
        elif file.name.endswith('.xlsx'):
            df=pd.read_excel(file)
            st.success('File XLSX caricato correttamente')
        else:
            st.error("Attenzione! Errore caricamento file. Formato non supportato")
        
        df1=df.copy()
        df1.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

        st.dataframe(df1)
        st.text('')
        st.text("Vediamo com'è composto il dataset")
        df1.describe().T
        st.text('')
        st.text("Qui è possibile vedere il Balancing del Dataset")
        st.write(df1['class'].value_counts())
        st.markdown("## Pairplot")
        fig=sns.pairplot(df1, hue='class', height=2)
        st.pyplot(fig)
        st.text("")
        st.markdown("## Valori nulli")
        st.write(df1.isnull().sum())
        st.markdown("## Grafico a Violino")
        sepal_length=df1['sepal_length']
        sepal_width=df1['sepal_width'] 
        petal_length=df1['petal_length'] 
        petal_width=df1['petal_width']
        st.multiselect('select', options=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        fig2=sns.violinplot(x=st.multiselect(), y=st.multiselect())
        #st.pyplot()
        # numerical_features = [x for x, dtype in zip(X.columns, X.dtypes) if dtype.kind in ['i','f'] ]
        # categorical_features = [x for x, dtype in zip(X.columns, X.dtypes) if dtype.kind not in ['i','f']]

        


if __name__ == "__main__":
    main()   

