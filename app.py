import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io 

st.set_page_config(page_icon="random")

def convert_to_excel(df_fea):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df_fea.to_excel(writer, sheet_name="data",index=False)
    # see: https://xlsxwriter.readthedocs.io/working_with_pandas.html
    writer.close()
    return output.getvalue()

def main():
    st.title('Progetto di Data Analysis di Livio Forza')
    st.text("Scegli un file da caricare che sia Excel o CSV")
    file= st.file_uploader("Carica", type=['csv', 'xlsx',])
    if file is not None:
        if file.name.endswith('.csv'):
            df=pd.read_csv(file, header=None)
            st.success('File CSV caricato correttamente!')
            st.balloons()
        elif file.name.endswith('.xlsx'):
            df=pd.read_excel(file)
            st.success('File XLSX caricato correttamente')
            st.balloons()
        else:
            st.error("⚠️ Attenzione! Errore caricamento file. Formato non supportato")
        
        df1=df.copy()
        df1.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        st.dataframe(df1)
        
        
    tab1, tab2, tab3 = st.tabs(["Preprocessin & EDA 🗃", "Prediction 🍟", "Batch 🎏"])
    with tab1:
        if st.button('Statistiche', help="Process Dataframe"):
            st.subheader('📈 Statistiche Descrittive')
            st.dataframe(df1.describe().T)

        if st.button('Grafici', help="Process Dataframe"):
            st.subheader('🎨 Pairplot delle Variabili Selezionate')
            fig = sns.pairplot(df1, hue='class', diag_kind='kde',height=4)
            st.pyplot(fig)

    with tab2:
        if st.button('Inferenza', help="Process Dataframe"):
            st.subheader(":magic_wand: Inferenza")

        sepal_length = st.slider('sepal length', 0.0, 9.0, 2.5)
        sepal_width = st.slider('sepal width', 0.0, 7.0, 2.5)
        petal_length = st.slider('petal length', 0.0, 9.0, 2.5)
        petal_width = st.slider('petal width', 0.0, 4.0, 2.5)
    
        data = {
            'sepal_length': [sepal_length],
            'sepal_width': [sepal_width],
            'petal_length': [petal_length],
            'petal_width': [petal_width]
            }

        pred_data = pd.DataFrame(data)

        iris_pipe = joblib.load('Iris_pipe.pkl')

        if st.button('Prediction'):
            pred = iris_pipe.predict(pred_data).astype(int)[0]
            classes = {0: 'SETOSA',
                    1: 'VERSICOLOR',
                    2: 'VIRGINICA'}
            y_pred = classes[pred]
            st.success(y_pred)

    with tab3:
        st.subheader("Adesso facciamo il baching")
        file2= st.file_uploader("Carica", type=['csv', 'xlsx'], key=1)
        if file2 is not None:
            if file2.name.endswith('.csv'):
                df_fea=pd.read_csv(file2)
                st.success('File CSV caricato correttamente!')
                st.balloons()
            elif file2.name.endswith('.xlsx'):
                df_fea=pd.read_excel(file2)
                st.success('File XLSX caricato correttamente')
                st.balloons()
            else:
                st.error("⚠️ Attenzione! Errore caricamento file. Formato non supportato")
            
                df_fea.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        
        if st.button('Start Processing', help="Process Dataframe"):
            st.header('Addes Column')
            y_pred_tot = iris_pipe.predict(df_fea)
            
            df_fea['class_pred'] = y_pred_tot
            st.dataframe(df_fea)
            st.balloons()
            st.download_button(
                                label="download as Excel-file",
                                data=convert_to_excel(df_fea),
                                file_name="data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="excel_download",
                                )

if __name__ == "__main__":
    main()


