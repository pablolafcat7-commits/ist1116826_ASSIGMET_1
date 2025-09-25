%Dataset 1: Diabetes Dataset (Regression)

datos= readtable("diabetes.tab.txt");
x= datos{:,1:10};   %Predictor variables
y= datos{:,11};   %Tag (0/1)

cv=cvpartition(y,'HoldOut',0.3); %Division de datos (70% entrenamiento, 30% prueba)
xtrain=x(training(cv),:); %datos entrenamiento x
ytrain=y(training(cv),:); %etiquetas entrenamiento y
xtest=x(test(cv),:); %datos de prueba x
ytest=y(test(cv),:); %etiquetas de prueba y


%%REGRESION LINEAL 

lineal=fitglm(xtrain,ytrain);  %Modelo de regresion lineal
ypredic = predict(lineal,xtest);
rmse = sqrt(mean((ytest - ypredic).^2));
r2 = corr(ytest,ypredic)^2;

disp('Modelo lineal');
disp(['RMSE: ', num2str(rmse)]);
disp(['R2: ', num2str(r2)]);

%%FUZZY MODEL ANFIS   
trainData = [xtrain ytrain];
testData  = [xtest ytest];

numMFs = 2;
mfType = 'gbellmf';  %Gauss Bell
inFIS = genfis1(trainData,numMFs,mfType,'linear');

epoch_n = 2;   %few iterations since otherwise it would take a long time to compile
[trnFIS,trnError,~,chkFIS,chkError] = anfis(trainData,inFIS,epoch_n,[],testData);

ypred_fuzzy = evalfis(xtest,chkFIS);
rmse_fuzzy = sqrt(mean((ytest - ypred_fuzzy).^2));
r2_fuzzy = corr(ytest,ypred_fuzzy)^2;

disp('Fuzzy Model ANFIS');
disp(['RMSE_ANFIS: ', num2str(rmse_fuzzy)]);
disp(['R2_ANFIS: ', num2str(r2_fuzzy)]);


%Model Comparation

figure; clf;
plot([trnError chkError],'LineWidth',1.5);
legend('Entrenamiento','Validación');
xlabel('Épocas'); ylabel('Error RMSE');
title('Curva de aprendizaje ANFIS');

figure;
scatter(ytest,ypredic,'b'); hold on;
scatter(ytest,ypred_fuzzy,'r');
plot([min(ytest) max(ytest)],[min(ytest) max(ytest)],'k--');
legend('Regresión lineal','Fuzzy ANFIS','Ideal');
xlabel('Valor real'); ylabel('Predicción');
title('Comparación modelos de regresión');
grid on;