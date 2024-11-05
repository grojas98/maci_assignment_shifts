# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 01:47:56 2024

@author: groja
"""

import streamlit as st
import pandas as pd
import pickle
from streamlit_autorefresh import st_autorefresh
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
def load_model():
    # Cargar los datos (ajusta el nombre si es necesario)
    df = pd.read_csv('dbs/data.csv',sep=';')

    # Revisar los primeros datos
    df.head()
    # Convertir variables categóricas en variables dummies
    df = pd.get_dummies(df, columns=['tipo_profesional', 'tipo_contrato', 'turno_asignado', 'opcion_turno'])

    # Separar las características (X) y la variable objetivo (y)
    X = df.drop(['elegible_turno', 'id', 'nombre'], axis=1)
    y = df['elegible_turno']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_rf_model = RandomForestRegressor(max_depth=10, max_features='auto', n_estimators=50,
                          random_state=42)
    best_rf_model.fit(X_train, y_train)
    
    return best_rf_model

def turnos():

    if "rol" not in st.session_state:
        st.session_state.rol = ''
    if "turno" not in st.session_state:
        st.session_state.turno = ''
    if "df" not in st.session_state:
        st.session_state.df = ''

    final_df = pd.DataFrame()
    profesionales = ['auxiliar', 'matrona', 'enfermero', 'kinesiólogo']
    contratos = ['contratado', 'reemplazo']
    turnos_asignados = ['4to_turno', '3er_turno', 'diurno', 'no']
    opciones_turno = ['4to_turno', '3er_turno', 'diurno']
    
    st.subheader("Ingresar turno disponible")
    st.caption("Seleccionar información del turno disponible debido a licencia/vacaciones para la asignación automática.")
    
    best_rf_model = load_model()

    st.session_state.df = pd.read_excel('prototipo_turnos.xlsx')    
    del st.session_state.df['elegible_turno']

    form_t = st.empty()
    form_t = form_t.form(key='form_update2')
    
    st.session_state.rol = form_t.selectbox(
        'Tipo de profesional:',
        ['Enfermera', 'Auxiliar', 'Kinesiólogo', 'Matrona'],key='select1')
    
    st.session_state.turno = form_t.selectbox(
        'Turno liberado:',
        ['4to Turno', '3er Turno', 'Diurno'],key='select2')

    submit_button_update = form_t.form_submit_button(label='Ingresar')
    
    if submit_button_update:
        with st.spinner("Asignando, espera un poquito..."):
            time.sleep(3)
            st.session_state.rol = st.session_state.rol.lower().replace('enfermera','enfermero')
            st.session_state.turno = st.session_state.turno.lower().replace(' ','_')
            df_v2 = st.session_state.df.loc[st.session_state.df['tipo_profesional']==st.session_state.rol] 
            df_v3 = df_v2.loc[df_v2['opcion_turno']==st.session_state.turno] 
            
            
            turnos_planta_v3 = df_v3.reset_index(drop=True)
            for i in range(len(turnos_planta_v3)):
                row = turnos_planta_v3.iloc[[i]]
                nombre = row['nombre'].unique()[0]
                id_ = row['id'].unique()[0]
                edad = row['edad'].unique()[0]
                turno_asignado = row['turno_asignado'].unique()[0]
                turno = turnos_planta_v3['opcion_turno'].unique()[0]
                antiguedad = row['antiguedad'].unique()[0]
                preferencia = row['preferencia'].unique()[0]
                tipo_profesional = row['tipo_profesional'].unique()[0]
                tipo_contrato = row['tipo_contrato'].unique()[0]
                row_v2 = pd.get_dummies(row, columns=['tipo_profesional', 'tipo_contrato', 'turno_asignado', 'opcion_turno'])
                for prof in profesionales:
                    if prof in row['tipo_profesional'].unique():
                        pass
                    else:
                        row_v2.loc[:,f'tipo_profesional_{prof}'] = 0
                for con in contratos:
                    if con in row['tipo_contrato'].unique():
                        pass
                    else:
                        row_v2.loc[:,f'tipo_contrato_{con}'] = 0
                        
                for turn in turnos_asignados:
                    if turn in row['turno_asignado'].unique():
                        pass
                    else:
                        row_v2.loc[:,f'turno_asignado_{turn}'] = 0
                        
                for opc in opciones_turno:
                    if opc in row['opcion_turno'].unique():
                        pass
                    else:
                        row_v2.loc[:,f'opcion_turno_{opc}'] = 0
                
                
                del row_v2['id']
                del row_v2['nombre']
                
                row_v3 = row_v2[['edad', 'antiguedad', 'preferencia',
                       'tipo_profesional_auxiliar', 'tipo_profesional_enfermero', 'tipo_profesional_kinesiólogo',
                       'tipo_profesional_matrona', 'tipo_contrato_contratado',
                       'tipo_contrato_reemplazo', 'turno_asignado_3er_turno',
                       'turno_asignado_4to_turno', 'turno_asignado_diurno',
                       'turno_asignado_no', 'opcion_turno_3er_turno',
                       'opcion_turno_4to_turno', 'opcion_turno_diurno']]
                
                prediccion = best_rf_model.predict(row_v3)
                print(f"Predicción de elegible_turno para el nuevo profesional: {prediccion[0]}")
                result = pd.DataFrame({
                    'id': [id_],
                    'nombre': [nombre],
                    'edad': [edad],
                    'tipo_contrato': [tipo_contrato],
                    'tipo_profesional': [tipo_profesional],
                    'antiguedad': [antiguedad],
                    'turno_asignado':[turno_asignado],
                    'turno': [turno],
                    'preferencia': [preferencia],
                    'prediccion': [prediccion[0]]
                    })
                
                final_df = pd.concat([final_df,result])
            if not final_df.empty:
                final_df = final_df.sort_values(by='prediccion',ascending=False)
                final_df = final_df.reset_index(drop=True)
                
                if all(s == 0 for s in final_df['prediccion']):
                    st.warning("No hay profesionales aplicables para cubrir el turno.")
                    st.dataframe(final_df)
                else:
                    st.success(f"Turno aplicable para profesional: {final_df['nombre'][0]}")
                    st.dataframe(final_df.style.applymap(
                        lambda _: "background-color: LightGreen;", subset=([0], slice(None))
                    ))
                    
                    st.balloons()
            else:
                st.warning("No hay profesionales disponibles para cubrir el turno.")
            
    
def database():
    st.session_state.df = pd.read_excel('prototipo_turnos.xlsx')  
    del st.session_state.df['elegible_turno']
    st.dataframe(st.session_state.df)


def main(): # COMENTARIOS: AÑADIR INFORMES PENDIENTES | KINESIC CARE: KINESIOLOGO, TO, FONO / PUERTO PARAÍSO: TENS | CREDENCIALES POR PRESTADOR
    # --- Initialising SessionState ---
    if "state1" not in st.session_state:
        st.session_state.state1 = 0
    if "state_but" not in st.session_state:
        st.session_state.state_but = 0
    if "changes" not in st.session_state:
        st.session_state.changes = 0
    if 'agenda_state' not in st.session_state:
        st.session_state.agenda_state = {

            'df': None,
            'backup':None,
            'agenda': None

        }
    if st.session_state.state1 == 0:
        
        col1, mid, col2 = st.columns([0.5,1.8,0.5])
        # with col1:
        #     image = st.image('images/at.png', width=150)
        with mid:
            st.write("")
            st.write("")
            title = st.markdown("<h1 style='text-align: center; color: black;'>Asignación inteligente de turnos</h1>", unsafe_allow_html=True)
        # with col2:
        #     image = st.image('images/logo_ht.png', width=180)
        st.caption("¡Asigna turnos de manera inteligente con la ayuda de la IA!")
        form_ = st.empty()
        form_ = form_.form(key='my_form_42')
        # if st.session_state.blocked == 0:
        menu = form_.selectbox(
            'Seleccionar interfaz a visualizar:',
            ('Ingresar turno nuevo','Ver base de datos','DEV'),key=13)
        # else:
        #     menu = form_.selectbox(
        #         'Seleccionar interfaz a visualizar:',
        #         ('Ingresar agenda','Modificar agenda','Ver agenda'),key='blocked_menu') # añadir restricciones de autorización 
        
        submit_button = form_.form_submit_button(label='Ingresar a interfaz')
        if submit_button:

            DEL_SESSIONS = ['update_df_2','df_2','df_v2_2','state_update_2','update_drive','update_df','able_to_change','state_update','df',
                            'df_v2','df_v3','df_v4','df_v5','att_terminated','init_n','paciente','tipo_profesional','profesional','n_atenciones',
                            'fecha','turno','date_list','turnos_list','prof_list','rut_prof_list','init_date','filters_mod','eliminar_state',
                            'update_df_3','df_3','filters_mod2','calendar_events','instructions_dict','id_count','date_pick','refresh_status',
                            'event_change','state_update_3','paciente_agenda','periodo','prof_ver','calendar_events2']
            
            for s in DEL_SESSIONS:
                try:
                    del st.session_state[s]
                except:
                    continue
        if submit_button or st.session_state.state_but == 1:
            
            st.session_state.state_but = 1
                
            if menu == 'Ingresar turno nuevo':

                turnos()
                
            elif menu == 'Ver base de datos':
                database() 

            

        if st.button('Cerrar sesión'):

            st.session_state.state2 = 0
            st.session_state.state_but = 0
            st.info('Cerrando sesión...')
            st.cache_data.clear()
            DEL_SESSIONS = ['update_df_2','df_2','df_v2_2','state_update_2','update_drive','update_df','able_to_change','state_update','df',
                            'df_v2','df_v3','df_v4','df_v5','att_terminated','init_n','paciente','tipo_profesional','profesional','n_atenciones',
                            'fecha','turno','date_list','turnos_list','prof_list','rut_prof_list','init_date','prestador','filters_mod','eliminar_state',
                            'update_df_3','df_3','filters_mod2','blocked','calendar_events','instructions_dict','id_count','date_pick','refresh_status',
                            'event_change','reload_id','state_update_3','paciente_agenda','periodo','prof_ver','calendar_events2']
            
            for s in DEL_SESSIONS:
                try:
                    del st.session_state[s]
                except:
                    continue

            st_autorefresh()

#%% LLAMAR A FUNCIÓN LOGIN

if __name__ == '__main__':
    main()


