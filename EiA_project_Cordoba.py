## Alliance of International Bioversity and CIAT
## December, 2023
## Creator: Maria Victoria Díaz 
## Codes based on Hugo Dorado job

import warnings 
import os
import glob
import re
import pandas as pd 
import numpy as np
import sklearn 
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import csv
import fnmatch
import catboost as cb
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from catboost import *
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import random as rd
from random import sample
import dask
import dask.dataframe as dd
from dask import delayed as dy
from dask import compute
from concurrent.futures import ThreadPoolExecutor
dask.config.set(scheduler='processes')
dask.config.set(num_workers=10, processes=True)

warnings.filterwarnings ("ignore")
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class EiA_Cordoba():

    def __init__(self,path):
        self.path = path
        self.path_data = os.path.join(self.path,'datos')
        self.path_models = os.path.join(self.path,'modelos')
        self.path_resamp = os.path.join(self.path,'datos','resampling')

        pass

    #def extract_escenarios(self):
    #falta que hagan disponibles los escenarios desde la api de aclimate
  
    def calculo_indicador(self, climDB, stageIni, stageEnd, namFun):
    
        stageIni = stageIni.iloc[0]
        stageEnd = stageEnd.iloc[0]

        if stageIni > stageEnd:
            
            climDB1 = climDB.loc[(climDB['DATE'] >= stageIni)]
            climDB2 = climDB.loc[(climDB['DATE'] <= stageEnd)]

            climDB = pd.concat([climDB1, climDB2], axis=0, ignore_index=True)
        
        else:
        
            climDB = climDB.loc[(climDB['DATE'] >= stageIni) & (climDB['DATE'] <= stageEnd)]
            print(climDB['RAIN'].sum())

        if climDB.empty:
                        
                return None

        else:
                
                media_max = climDB['TMAX'].mean()
                media_min = climDB['TMIN'].mean()
                media_max_plus_min = (climDB['TMAX'] + climDB['TMIN'])/2
                media_max_plus_min = media_max_plus_min.mean()
                media_max_minus_min = (climDB['TMAX'] -climDB['TMIN'])
                media_max_minus_min = media_max_minus_min.mean()
                sum_eso = climDB['ESOL'].sum()
                prom_tmax_34 = (climDB.loc[climDB['TMAX'] >34, 'TMAX'].sum())/len(climDB['TMAX'])
                sum_rain = climDB['RAIN'].sum()
                prom_rain_10= (climDB.loc[climDB['RAIN'] >10, 'RAIN'].sum())/len(climDB['RAIN'])
                prom_tmin_15= (climDB.loc[climDB['TMIN'] <15, 'TMIN'].sum())/len(climDB['TMIN'])

                clIndFin = pd.DataFrame([[media_max, media_min, media_max_plus_min, media_max_minus_min, sum_eso, prom_tmax_34, sum_rain, prom_rain_10, prom_tmin_15]], columns = namFun)
                return clIndFin
    

    def escenarios(self, csvs, cosechBase,namFun, escenario,state, dir_save_ind):
    
      
        print('INICIANDO PROCESO PARA EL ESCENARIO '+ str(escenario) )

        climBase = pd.read_csv(csvs)
        climBase.columns = ["day","month","year","TMAX","TMIN","RAIN","ESOL"]
        climBase['DATE'] = climBase['month'].astype(str) + '/' + climBase['day'].astype(str) #+ '/' + climBase['year'].astype(str)
        climBase['DATE'] = pd.to_datetime(climBase['DATE'], format = '%m/%d')
        climBase = climBase.drop(columns=['day', 'month', 'year'])
        cosechBase = cosechBase.reset_index()
        cbase = cosechBase.copy()

        cbase = cbase.drop(['FECHA_SIEMBRA_NEW', 'FECHA_COSECHA_NEW', 'FECHA_FLORACION_NEW', 'FECHA_EMERGENCIA_NEW'], axis = 1)    
        base = []
        Indicadores_escenario = []


        for i in range(len(cosechBase)):
            print(cosechBase.loc[i,'ID_LOTE'])
           

            indcStag1 = pd.DataFrame(self.calculo_indicador(climBase, cosechBase.loc[i]['FECHA_EMERGENCIA_NEW'], cosechBase.loc[i]['FECHA_FLORACION_NEW'], namFun))
            indcStag2 = pd.DataFrame(self.calculo_indicador(climBase,  cosechBase.loc[i]['FECHA_FLORACION_NEW'],cosechBase.loc[i]['FECHA_COSECHA_NEW'], namFun)) #mad[i]


            if (indcStag1.empty) | (indcStag2.empty):
                  
                  Indicadores_escenario = Indicadores_escenario
                  base = base

            else:                 
                

                indcStag1.columns =indcStag1.columns + '_' + state[0]     
                indcStag2.columns = indcStag2.columns + '_' + state[1]

                df = pd.concat([indcStag1.reset_index(drop=True), indcStag2], axis=1)   
                        
                Indicadores_escenario.append(df)
                base.append(pd.DataFrame(cbase.iloc[i, :]).transpose())


        print('GUARDANDO ESCENARIO...')
        
        df = pd.concat(Indicadores_escenario)
        base = pd.concat(base)
        df_c = pd.concat([base.reset_index(drop=True), df.reset_index(drop = True)], axis = 1)
        df_c.drop(['index'], axis = 1)

        df_c.to_csv(os.path.join(dir_save_ind, f"escenario_{str(escenario)}.csv"), index =False)
        print('ESCENARIO '+ str(escenario) + ' GUARDADO')
        

        return df_c
    
    def indicadores_definitivos(self, id_estacion, cosechBase_filename, cercania_fincas_estaciones_fn,state, namFun,  dir_save):
    
        cosechBase = pd.read_csv(cosechBase_filename)

            
        print('INICIANDO PROCESO PARA LA ESTACIÓN ' + id_estacion + '...')

        cercania_finca_estacion = pd.read_csv(cercania_fincas_estaciones_fn)
        fincas_id = cercania_finca_estacion.loc[cercania_finca_estacion['ID_ESTACION'] == id_estacion, 'ID_LOTE']

        if fincas_id is None:

                print('La estación no es cercana a algún lote')

        else:    

                cosechBase = cosechBase.loc[cosechBase['ID_LOTE'].isin(fincas_id)]


                if cosechBase.empty:

                    print('Fincas no encontradas por fechas erróneas')


                else:

                    cosechBase['FECHA_SIEMBRA'] = pd.to_datetime(cosechBase['FECHA_SIEMBRA'], format ="%m/%d/%Y" )
                    cosechBase['FECHA_SIEMBRA_NEW'] = cosechBase['FECHA_SIEMBRA'].dt.month.astype(str) + "/" + cosechBase['FECHA_SIEMBRA'].dt.day.astype(str)
                    cosechBase['FECHA_SIEMBRA_NEW'] = pd.to_datetime(cosechBase['FECHA_SIEMBRA_NEW'], format ="%m/%d" )



                    cosechBase['FECHA_COSECHA'] = pd.to_datetime(cosechBase['FECHA_COSECHA'], format ="%m/%d/%Y" )
                    cosechBase['FECHA_COSECHA_NEW'] = cosechBase['FECHA_COSECHA'].dt.month.astype(str) + "/" + cosechBase['FECHA_COSECHA'].dt.day.astype(str)
                    cosechBase['FECHA_COSECHA_NEW'] = pd.to_datetime(cosechBase['FECHA_COSECHA_NEW'], format ="%m/%d" )



                    cosechBase['FECHA_FLORACION'] = pd.to_datetime(cosechBase['FECHA_FLORACION'], format ="%m/%d/%Y" )
                    cosechBase['FECHA_FLORACION_NEW'] = cosechBase['FECHA_FLORACION'].dt.month.astype(str) + "/" + cosechBase['FECHA_FLORACION'].dt.day.astype(str)
                    cosechBase['FECHA_FLORACION_NEW'] = pd.to_datetime(cosechBase['FECHA_FLORACION_NEW'], format ="%m/%d" )


                    cosechBase['FECHA_EMERGENCIA'] = pd.to_datetime(cosechBase['FECHA_EMERGENCIA'], format ="%m/%d/%Y" )
                    cosechBase['FECHA_EMERGENCIA_NEW'] = cosechBase['FECHA_EMERGENCIA'].dt.month.astype(str) + "/" + cosechBase['FECHA_EMERGENCIA'].dt.day.astype(str)
                    cosechBase['FECHA_EMERGENCIA_NEW'] = pd.to_datetime(cosechBase['FECHA_EMERGENCIA_NEW'], format ="%m/%d" )

                    dir_save_ind = os.path.join(dir_save, id_estacion, 'indicadores')

                    if not os.path.exists(dir_save_ind):
                        os.mkdir(dir_save_ind)


                    dir_list = glob.glob(os.path.join(dir_save, id_estacion, "\*.csv"))

                    esc = [x.replace("\\",'/') for x in dir_list]
                    esc = [re.sub(dir_save + '/'+id_estacion+'/'+id_estacion+'_escenario_', '', x) for x in esc]
                    esc =  [re.sub('.csv', '', x) for x in esc]


                    df_c = [self.escenarios(csvs =a,
                                cosechBase = cosechBase, 
                                namFun=namFun, 
                                state =state, 
                                escenario = str(b), dir_save_ind= dir_save_ind) for a,b in zip(dir_list, esc) ]
                                            

                        

                    return(df_c)
    


    def generador(self, namFun):

   
        estaciones = os.listdir(self.path_resamp)
  
        for i in range(0, len(estaciones)):
            print(estaciones[i])
            self.indicadores_definitivos(id_estacion = estaciones[i],
                                    cosechBase_filename = os.path.join(self.path_data, 'Data_Cordoba_final.csv'), 
                                    cercania_fincas_estaciones_fn =  os.path.join(self.path_data,'cercania_fincas.csv'), 
                                    state = ['Veg',  'Rep'], namFun = namFun,  dir_save = self.path_data)
        
            

    def climate_base(self, state, namFun):
    
        dataset = pd.read_csv(os.path.join(self.path_data, 'Data_Cordoba_final.csv'))


        daily = pd.read_csv(os.path.join(self.path_data,'clima_daily_data.csv'))
        daily['DATE'] = daily['month'].astype(str) + "/" + daily["day"].astype(str) + "/" + daily["year"].astype(str)
        daily = daily.drop(columns=['day', 'month', 'year'])
        daily['DATE'] = pd.to_datetime(daily['DATE'], format ="%m/%d/%Y" )
        dataset['FECHA_EMERGENCIA'] = pd.to_datetime(dataset['FECHA_EMERGENCIA'], format ="%m/%d/%Y" )
        dataset['FECHA_FLORACION'] = pd.to_datetime(dataset['FECHA_FLORACION'], format ="%m/%d/%Y" )
        dataset['FECHA_COSECHA'] = pd.to_datetime(dataset['FECHA_COSECHA'], format ="%m/%d/%Y" )


        ind = []

        for i in dataset['ID_EVENTO']:

            d = daily[daily['ID_EVENTO'] == i]    

            
            indcStag1 = pd.DataFrame(self.calculo_indicador(climDB = d, stageIni=dataset[dataset['ID_EVENTO'] == i]['FECHA_EMERGENCIA'],stageEnd=dataset[dataset['ID_EVENTO'] == i]['FECHA_FLORACION'], namFun=namFun))
            indcStag2 = pd.DataFrame(self.calculo_indicador(climDB = d , stageIni=dataset[dataset['ID_EVENTO'] == i]['FECHA_FLORACION'],stageEnd=dataset[dataset['ID_EVENTO'] == i]['FECHA_COSECHA'], namFun=namFun)) #mad[i]

            indcStag1.columns =indcStag1.columns + '_' + state[0]     
            indcStag2.columns = indcStag2.columns + '_' + state[1]

                
            df = pd.concat([indcStag1.reset_index(drop=True), indcStag2], axis=1) 

            ind.append(df)
        
        ind = pd.concat(ind)

        df1 = pd.concat([dataset, ind.reset_index(drop=True)], axis = 1)  

        df1.to_csv(os.path.join(self.path_data, 'Data_Cordoba_Final_Clima.csv'), index = False)
                            
        return df1
    
    # ML models

    def params_rf(self):

        params = {'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'n_estimators': [10, 50, 100, 200, 300,400,500,1000]}
        

        gd_rf = GridSearchCV(estimator  = RandomForestRegressor(random_state=123),
                        param_grid=params,
                        scoring    = 'neg_root_mean_squared_error',
                        n_jobs = -1,
                        cv=4)

        p = ColumnTransformer([
        ('scale', MinMaxScaler(),
        make_column_selector(dtype_include=np.number))
        ],remainder = OneHotEncoder(handle_unknown='ignore'), verbose_feature_names_out=False)

        return([gd_rf,  p])
    
        
    def models(self, model_type, test_size = 0.30):

        dataset = pd.read_csv(os.path.join(self.path_data, 'Data_Cordoba_Final_Clima.csv'))
        dataset = dataset.dropna()

        Y = dataset["RDT"]
        X1 = dataset.drop(["RDT", "ID_LOTE","FECHA_SIEMBRA","FECHA_EMERGENCIA", "FECHA_FLORACION","FECHA_COSECHA", 'ID_EVENTO', 'LAT_LOTE', 'LONG_LOTE'], axis = 1)

        if model_type == "cb":
        
            X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size= test_size, random_state=101)

            
            categorical_columns = X1.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_features_indices = [X1.columns.get_loc(col) for col in categorical_columns]

            model = CatBoostRegressor(loss_function='RMSE',  cat_features=categorical_features_indices)

            model.fit(X_train, y_train,  early_stopping_rounds=50, verbose=100)

            
            if not os.path.exists(self.path_models):
                os.mkdir(self.path_models)
        
            model.save_model(os.path.join(self.path_models,'catboost_cordoba.cbm'))

            return ([model, X_test, y_test, X_train, y_train])


        else: 
            
            model, preprocessor = self.params_rf()

            X_train_prep = preprocessor.fit_transform(X_train)
            X_test_prep  = preprocessor.transform(X_test)
        

            model.fit(X_train_prep, y_train)


            if not os.path.exists(self.path_models):
                os.mkdir(self.path_models)
        
            dump(preprocessor, os.path.join(self.path_models, 'pipe_line.joblib') )
            dump(model, os.path.join(self.path_models,'rf_cordoba.joblib'))
                
            return ([model, X_test_prep, y_test, X_train_prep, y_train])
        
        
    def importance_graph(self,modelo, model_type, top_n):

        if model_type == "cb":

            m = modelo[0]
            feature_importance = m.feature_importances_
            sorted_idx = np.argsort(feature_importance)[-top_n:]
            plt.figure(figsize=(10, 9))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(modelo[1].columns)[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Most Important Features')
            plt.savefig(os.path.joi(path ,'features_importance.png'), dpi=600)
            
        else:

            m = modelo[0].best_estimator_
            p = self.params_rf()

            # Get feature importances from the best model
            feature_importances = m.feature_importances_

            # Get the original feature names from the ColumnTransformer
            numeric_feature_names = p[1].transformers_[0][2]

            # Create a dictionary to map feature names to their importance scores
            feature_importance_dict = dict(zip(numeric_feature_names, feature_importances))

            # Sort the features by importance in descending order
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=False)

            # Get the top N features and their importances
            top_features = sorted_features[:top_n]
            feature_names, importances = zip(*top_features)

            # Create a bar plot
            plt.figure(figsize=(10, 6))
            plt.barh(range(top_n), importances, align='center')
            plt.yticks(range(top_n), feature_names)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Most Important Features')
            plt.savefig(os.path.join(self.path,'features_importance.png'), dpi=600)



     # Optimization algorithm
    
     ## Human sort
    def atoi(self, text):
        '''
        Function to turn  text into integer if it´s number
        
        '''
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ] #Retorna los textos que hay entre cada número, todos separados en un vector.


    def extrac_ranges(self, var,val):
        '''
        Función para extraer los rangos de las variables
        
        '''
        if val == "Continuous" or val == "Discrete":
                max = var.apply(np.max)[0]
                min = var.apply(np.min)[0]
                ranges = [max,min]
        elif bool(re.search("Cat",val)):
                ranges = list(np.unique(var))
                ranges.sort(key=self.natural_keys) #Retorna los valores de las var categóricas, excluyendo los nros.
                #print(ranges)
        else:
                print("Error, unrecognized category")

        return(ranges)

     ## Validation

    def valDatasets(self, ds,reference):
        '''
        Función para validar que las variables de dos bases de datos sean las mismas
        y tengan la misma longitud
        
        '''
        if ds.shape[1] == reference.shape[0]:
            if sum(reference["Variable"] == ds.columns) == ds.shape[1]:
                message = "All values match"
            else:
                message ="Datasets and variables have different variables"
        else:
            message = "Datasets and variables have different lengths"
        return message


    def AllRangGen(self, dataset,scales,names):
        '''
        Función para extraer los rangos de las variables deseadas
        '''
        
        allRanges = []
        names = names.reset_index()["Variable"]
        scales = scales.reset_index()["Scale"]
        for i in range(0,dataset.shape[1]):
            variable = dataset.filter([names[i]])
            allRanges.append(self.extrac_ranges(variable,scales[i]))
            #print(i)
        return allRanges



    def RandomIni(self,rang,scale):
        '''
        Generar un número aleatorio dentro del rango de la variable 
        
        '''
        # por qué no generarlo desde la varaiabled misma?
        
        if scale == "Discrete":
            value = sample(range(round(rang[1]),(round(rang[0])+1)),1)
        elif scale == "Continuous":
            value = np.random.uniform(rang[1],rang[0],(1,))
        elif scale == "Category":
            value = sample(rang,1)
        else:    
            print("Unrecognized scale")
        return value[0]



    def RandomIniVec(self, RangosCompletos, escalasCompletas):
        '''
        Generar un valor aleatorio entre cada rango de variables
        y guardar esos números en un vector
        
        '''
        
        long_rang = len(RangosCompletos)   
        gen = []
        for i in range(0,long_rang):
            gen.append(self.RandomIni(RangosCompletos[i], escalasCompletas[i]))
            
        return gen


    def RandomPopIni(self, NumbPobIni,namesds,ds_ranges,scales):  
        '''
        Guardar el vector de valores generados en un dataframe y asignar los nombres a las columnas
        CONJUNTO DE SOLUCIONES INICIALES . EXTRAER N SAMPLES 
        '''
        df = pd.DataFrame(columns = namesds)
        
        for j in range(0,NumbPobIni):
            
            df.loc[j] = self.RandomIniVec(ds_ranges, scales)
        
        
        df.loc[df['TIPO_PREP'] == "Ninguna (siembra directa)", 'NUM_PASES_PREP' ] = 0
        df.loc[df['TIPO_MATERIAL'].isin(["Variedad", "Semilla Campesina"]) , 'MATERIAL_GENETICO' ] = "Otro"
        df.loc[df['MATERIA_ORGANICA'] == "BAJA", 'MATERIA_ORGANICA' ] = np.random.choice(["MEDIA", "ALTA"], len(df.loc[df['MATERIA_ORGANICA'] == "BAJA", 'MATERIA_ORGANICA']))
        mask = (df['PH'] < 5.5) | (df['PH'] > 6.5)
        df.loc[mask, 'PH'] = np.random.uniform(5.5, 6.5, len(df.loc[mask, 'PH']))
                
        return df

    def fitnessfun(self, model,inputs,fixedValues, columns):
        '''
        EVALUAR LA CALIDAD DE UNA SOLUCIÓN
        INPUTS = POSIBLES SOLUCIONES 
        '''
        
        inputs.reset_index(drop=True, inplace=True)
        fixedValues.reset_index(drop=True, inplace=True)
        mat = pd.concat([inputs,fixedValues],axis=1)
        return model.predict(mat[columns])


    def ImproviseFun(self, hm,hmcr,bestHarmony,par,ranges,scales):
        '''
        BUSCA OTRAS POSIBLES POTENCIALES SOLUCIONES A PARTIR DE LAS QUE YA TENGO
        '''

        hmss = hm.copy()
        del hmss["Performance"]
        harmony = pd.DataFrame( columns = hmss.columns)
        temp = []
        for i in range(0,hmss.shape[1]):
            if np.random.rand() < hmcr:
                xi = hm.sample(1, axis=0).iloc[0][i]
                if np.random.rand() < par:
                    xi = bestHarmony.iloc[0][bestHarmony.columns[i]]
            else:
                xi = self.RandomIni(ranges[i],scales[i])

            temp.append(xi)

        harmony.loc[0]= temp

        return harmony

    def bestGlobHS(self, fv,hms,hmcr,par,maxNumInp,namesds,model_train,ranges,scales, columns):
        '''
        fv = variables fijas. que no son de manejo
        hms = poblacion inicial. Hacer pruebas. A partir de cuántas soluciones
            maxnumpin = veces que se intenta mejorar el algoritmo
        '''   
        

        hm = self.RandomPopIni(hms,namesds,ranges,scales)# dataframe con valores aleatorios 
        

        popfv = pd.concat([fv]*(hms),ignore_index=True)#fv.append([fv]*(hms-1),ignore_index=True) #extender o reducir el rango 
        
        hm["Performance"] = self.fitnessfun(model = model_train,inputs=hm,fixedValues=popfv, columns = columns) #desempeño a cada solucion

        hm = hm.sort_values(by = "Performance")

        worsthm = hm.iloc[[0]]
        besthm  = hm.iloc[[hms-1]]

        best_sol = hm["Performance"]

        for j in range(hms,maxNumInp):
            newHM = self.ImproviseFun(hm,hmcr,besthm,par,ranges,scales)
        
            newHM["Performance"] = self.fitnessfun(model_train, newHM,fv, columns)
            #print(j,newHM["Performance"].iloc[0] ,"-",worsthm["Performance"].iloc[0])

            if newHM["Performance"].iloc[0] > worsthm["Performance"].iloc[0]:               
                hm = pd.concat([hm, newHM])#hm.append(newHM)
                
                hm = hm.sort_values(by = "Performance")

                hm = hm.iloc[[*range(1,(hms+1))]]

                besthm = hm.iloc[[hms-1]]
                worsthm = hm.iloc[[0]]
        

            best_sol[j] = besthm["Performance"].iloc[0]

        return pd.DataFrame(besthm)



    def get_solution(self,df, e, id_lote,other_variables,managment_vars,scales_managment_vars, model, ds_ranges, columns):
    
        dataset = df.dropna()

        print("Doing escenary " + str(e)) 
        
        dataset = dataset.drop(["RDT", "ID_LOTE","FECHA_SIEMBRA","FECHA_EMERGENCIA", "FECHA_FLORACION","FECHA_COSECHA", 'ID_EVENTO', 'LAT_LOTE', 'LONG_LOTE'], axis = 1, inplace = False)
        s = self.bestGlobHS(fv=dataset[other_variables], hms=5, hmcr=0.85, par=0.3, maxNumInp=50, 
                                            namesds=managment_vars,model_train=model, 
                                            ranges=ds_ranges, scales=scales_managment_vars, columns=columns)    
        
            
        s['ID_LOTE'] = id_lote
        
        print('Solutions for the scenary ready')

        return s   
    
    
    def calculate_ci(self, values):
        confidence_level = 0.95
        n = len(values)
        mean = np.mean(values)
        standard_error = stats.sem(values)
        margin_of_error = standard_error * stats.t.ppf((1 + confidence_level) / 2, n - 1)
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        return ('['+str(round(lower_bound))+", "+ str(round(upper_bound))+']')
    
      
    def process_group(self, s, numeric_cols, cate_cols):

        ci_by_group =s.groupby('ID_LOTE')['Performance'].agg(self.calculate_ci)
        median = pd.DataFrame(s.groupby('ID_LOTE')[numeric_cols].median().reset_index())
        mode_by_group =  s.groupby('ID_LOTE')[cate_cols].apply(lambda x: x.mode().iloc[0]).reset_index()
        data_final = pd.merge(median, mode_by_group,how='left', on='ID_LOTE')
        data_final['Yield'] = list(ci_by_group)
        return data_final
    
    def process_scen(self, id_lote):

        cercania_finca_estacion = pd.read_csv(os.path.join(self.path_data,'cercania_fincas.csv'))

        station_id = cercania_finca_estacion.loc[cercania_finca_estacion['ID_LOTE'] == id_lote]['ID_ESTACION'].astype(str)

        dataset = pd.read_csv(os.path.join(self.path_data,'Data_Cordoba_Final_Clima.csv')).reset_index()

        clasVars = pd.read_csv(os.path.join(self.path_data,"header_dataset.csv"))

        managment_vars = clasVars[clasVars.Type=="M"]["Variable"].reset_index()["Variable"]
        scales_managment_vars = clasVars[clasVars.Type=="M"]["Scale"].reset_index()["Scale"]

        other_variables = clasVars[clasVars.Type!="M" ]
        other_variables = other_variables[other_variables.Type != "O"]["Variable"].reset_index()["Variable"]
        mat_M = dataset[managment_vars]
        ds_ranges = self.AllRangGen(mat_M,scales_managment_vars,managment_vars) 

                
        print("Processing station " + station_id.iloc[0] )  


        if station_id is not None: 

            f = os.path_join(self.path_resamp, station_id.iloc[0], "indicadores")

            if os.path.exists(f):
                
                x = glob.glob(os.path.join(f, '*.csv')) 

                dataframes = [pd.read_csv(archivo) for archivo in x]
                selected_records = [df[df['ID_LOTE'] == id_lote] for df in dataframes]

                X1 = dataset.drop(['index',"RDT", "ID_LOTE","FECHA_SIEMBRA","FECHA_EMERGENCIA", "FECHA_FLORACION","FECHA_COSECHA", 'ID_EVENTO', 'LAT_LOTE', 'LONG_LOTE'], axis = 1)

                categorical_columns = X1.select_dtypes(include=['object', 'category']).columns.tolist()
                categorical_features_indices = [X1.columns.get_loc(col) for col in categorical_columns]

                data_columns = X1.columns.tolist()
                
                loaded_model = CatBoostRegressor(loss_function='RMSE',  cat_features=categorical_features_indices)
                model =  loaded_model.load_model(os.path_join(self.path_model, 'catboost_cordoba.cbm'))

                esc = [y.replace("\\",'/') for y in x]
                esc = [re.sub(f + '/'+'escenario_', '', x) for x in esc]
                esc =  [re.sub('.csv', '', x) for x in esc]

                solu = list(map(lambda z,y:  self.get_solution(z,y,id_lote, other_variables,managment_vars,scales_managment_vars, model, ds_ranges, data_columns), selected_records, esc))
                            
                print('Summarising....')

                s = pd.concat(solu).reset_index() 
                numeric_cols = s.select_dtypes(include=['float64', 'int']).columns.to_list()
                numeric_cols = [x for x in numeric_cols if x not in ['Performance', 'ID_LOTE', 'index']] 
                cate_cols = s.select_dtypes(include=['object']).columns.to_list()
        
                data_final = self.process_group(s, numeric_cols,cate_cols )

                if not os.path.exists(os.path.join(self.path_resamp,'recomendaciones')):
                    os.mkdir(os.path.join(self.path_resamp, 'recomendaciones'))

        
                data_final.to_csv(os.path.join(self.path_resamp, 'recomendaciones', "recomendacion_finca_",str(id_lote), '.csv'), index = False)# mode = 'a', index=False, header=not os.path.exists(f+'/recomendacion'))

    
                return data_final

        else:
                print('La estación que está cercana a su finca no ha sido analizada ')









            
                



            









            



                        


