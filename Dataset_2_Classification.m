%Dataset 2: Pima Indians Diabetes Dataset (Classification)

datos= readtable("diabetes.csv"); %Read the code
x= datos{:,1:8};   %Predictor variables
y=datos{:,9};      %Tag (0/1)

cv=cvpartition(y,'HoldOut',0.3); %Division de datos (70% entrenamiento, 30% prueba)
xtrain=x(training(cv),:); %datos entrenamiento x
ytrain=y(training(cv),:); %etiquetas entrenamiento y
xtest=x(test(cv),:); %datos de prueba x
ytest=y(test(cv),:); %etiquetas de prueba y

modelo=fitglm(xtrain,ytrain,'Distribution','Binomial');  %Modelo de regresion logistica

%Prediccion y accuracy
yprob = predict(modelo,xtest); 
yprediccion=round(yprob); %clasificacion 0 o 1
accuracy=mean(yprediccion==ytest); %calculo de aciertos

%Fuzzy model
fis = mamfis('Name','DiabetesClassifier');  %sistema mandami
inputs = {'Pregnancies',[0 30];'Glucose',[0 200];'BloodPressure',[0 122];'SkinThickness',[0 100];
    'Insulin',[0 846];'BMI',[0 70];'DiabetesPedigree',[0 2.5];'Age',[0 100]};  %entrada de datos de variables

for i = 1:size(inputs,1)
    name = inputs{i,1};
    range = inputs{i,2};
    fis = addInput(fis,range,'Name',name);
    %clasificacion low medium high
    fis = addMF(fis,name,'trimf',[range(1) range(1) (range(1)+range(2))/2],'Name','Low');
    fis = addMF(fis,name,'trimf',[range(1) (range(1)+range(2))/2 range(2)],'Name','Medium');
    fis = addMF(fis,name,'trimf',[(range(1)+range(2))/2 range(2) range(2)],'Name','High');
end
    
fis = addOutput(fis,[0 1],'Name','Diabetes');  %salida
fis = addMF(fis,'Diabetes','trimf',[-0.1 0 0.5],'Name','No');
fis = addMF(fis,'Diabetes','trimf',[0.5 1 1.1],'Name','Yes');
    
rules = [     
    2 0 0 0 0 0 0 0 1 1 1;  % Si Glucose=Medium entonces No
    3 0 0 0 0 0 0 0 2 1 1;  % Si Glucose=High entonces Sí
    0 0 0 0 0 2 0 0 2 1 1;  % Si BMI=Medium entonces Sí
    0 0 0 0 0 3 0 0 2 1 1;  % Si BMI=High entonces Sí
    0 0 0 0 0 0 0 3 2 1 1;  % Si Age=High entonces Sí
    3 0 0 0 0 3 0 0 2 1 1]; % Si Glucose=High Y BMI=High entonces Sí
%normas del modelo

fis = addRule(fis,rules);    
figure;
subplot(2,1,1); plotmf(fis,'input',8); title('Funciones de pertenencia - Age');
subplot(2,1,2); plotmf(fis,'input',1); title('Funciones de pertenencia - Pregnance');


ejemplo = [3 100 60 40 0 2 0 20]; %tras implementar el modelo introducimos 8 parametros 
out = evalfis(fis,ejemplo);
disp(['Salida difusa = ', num2str(out)]);
if out >= 0.5
    disp('Clasificación: Diabetes = Sí');
else
    disp('Clasificación: Diabetes = No');
end    
   
%Resultados test
C = confusionmat(ytest,yprediccion); %matriz de confusuon
vp = C(2,2); %verdaderos positivos
fp = C(1,2); %falsos positivos
fn = C(2,1); %falsos negativos
vn = C(1,1); %verdaderos negativos

precision = vp / (vp + fp);
recall = vp / (vp + fn);
F1 = 2 * (precision*recall) / (precision+recall);

disp('Resultados en test');
disp(['Exactitud:',num2str(accuracy)]);
disp(['Precision:',num2str(precision)]);
disp(['Recall:',num2str(recall)]);
disp(['Score_F1:',num2str(F1)]);
disp('Matriz de confusión:');
disp(C);

%Curva ROC y AUC
[~,~,~,AUC] = perfcurve(ytest, yprob, 1);
disp(['AUC:',num2str(AUC)]);

figure; clf;
hold on;
[Xroc,Yroc,~,AUC] = perfcurve(ytest, yprob, 1);
plot(Xroc,Yroc,'b-','LineWidth',2);
xlabel('Falsos Positivos'); 
ylabel('Verdaderos Positivos');
title(['Curva ROC (AUC = ', num2str(AUC),')']);
grid on;