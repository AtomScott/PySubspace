Search.setIndex({docnames:["contribution/contribution","examples/MNIST_example_with_SM","examples/index","examples/pca_to_sm","examples_scripts/index","examples_scripts/plot_ani","examples_scripts/plot_pca","examples_scripts/plot_subspace_pca","examples_scripts/plot_test","examples_scripts/sg_execution_times","examples_scripts/test","getting_started/installation","index","source/cvt","source/cvt.models","source/cvt.utils","source/modules","tutorials/CMSM","tutorials/GDA","tutorials/KMSM-KCMSM","tutorials/KPCA","tutorials/LDA","tutorials/MSM","tutorials/PCA","tutorials/SM","tutorials/eGDA","tutorials/getting_started","tutorials/index","tutorials/references","\u958b\u767a\u8005\u30e1\u30e2"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,sphinx:56},filenames:["contribution/contribution.md","examples/MNIST_example_with_SM.rst","examples/index.rst","examples/pca_to_sm.rst","examples_scripts/index.rst","examples_scripts/plot_ani.rst","examples_scripts/plot_pca.rst","examples_scripts/plot_subspace_pca.rst","examples_scripts/plot_test.rst","examples_scripts/sg_execution_times.rst","examples_scripts/test.rst","getting_started/installation.rst","index.rst","source/cvt.rst","source/cvt.models.rst","source/cvt.utils.rst","source/modules.rst","tutorials/CMSM.rst","tutorials/GDA.rst","tutorials/KMSM-KCMSM.rst","tutorials/KPCA.rst","tutorials/LDA.rst","tutorials/MSM.rst","tutorials/PCA.rst","tutorials/SM.rst","tutorials/eGDA.rst","tutorials/getting_started.rst","tutorials/index.rst","tutorials/references.rst","\u958b\u767a\u8005\u30e1\u30e2.md"],objects:{"":{cvt:[13,0,0,"-"]},"cvt.models":{base_class:[14,0,0,"-"],cmsm:[14,0,0,"-"],kcmsm:[14,0,0,"-"],kmsm:[14,0,0,"-"],msm:[14,0,0,"-"],sm:[14,0,0,"-"]},"cvt.models.base_class":{ConstrainedSMBase:[14,1,1,""],KernelCSMBase:[14,1,1,""],KernelSMBase:[14,1,1,""],MSMInterface:[14,1,1,""],SMBase:[14,1,1,""]},"cvt.models.base_class.ConstrainedSMBase":{param_names:[14,2,1,""]},"cvt.models.base_class.KernelCSMBase":{param_names:[14,2,1,""]},"cvt.models.base_class.KernelSMBase":{param_names:[14,2,1,""]},"cvt.models.base_class.MSMInterface":{predict_proba:[14,3,1,""],test_n_subdims:[14,3,1,""]},"cvt.models.base_class.SMBase":{fit:[14,3,1,""],get_params:[14,3,1,""],param_names:[14,2,1,""],predict:[14,3,1,""],predict_proba:[14,3,1,""],proba2class:[14,3,1,""],set_params:[14,3,1,""]},"cvt.models.cmsm":{ConstrainedMSM:[14,1,1,""]},"cvt.models.kcmsm":{KernelCMSM:[14,1,1,""]},"cvt.models.kmsm":{KernelMSM:[14,1,1,""]},"cvt.models.kmsm.KernelMSM":{fast_predict_proba:[14,3,1,""]},"cvt.models.msm":{MutualSubspaceMethod:[14,1,1,""]},"cvt.models.sm":{SubspaceMethod:[14,1,1,""]},"cvt.utils":{base:[15,0,0,"-"],evaluation:[15,0,0,"-"],kernel_functions:[15,0,0,"-"]},"cvt.utils.base":{canonical_angle:[15,4,1,""],canonical_angle_matrix:[15,4,1,""],canonical_angle_matrix_f:[15,4,1,""],cross_similarities:[15,4,1,""],dual_vectors:[15,4,1,""],max_square_singular_values:[15,4,1,""],mean_square_singular_values:[15,4,1,""],subspace_bases:[15,4,1,""]},"cvt.utils.evaluation":{calc_eer:[15,4,1,""],calc_er:[15,4,1,""]},"cvt.utils.kernel_functions":{l2_kernel:[15,4,1,""],linear_kernel:[15,4,1,""],rbf_kernel:[15,4,1,""]},cvt:{models:[14,0,0,"-"],utils:[15,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"0000ff":3,"11527923_8":28,"16th":28,"200m":5,"35s":1,"6_708":28,"\u03b8":3,"\u03d5i":[],"\u304c\u3044\u3044\u304b\u3082\u306d":29,"\u3053\u308c\u3044\u3089\u306a\u3044\u3088\u3046\u306b\u30b3\u30fc\u30c9\u5909\u3048\u308b":1,"\u3057\u306a\u304f\u3066\u3082\u4fdd\u5b58\u3059\u308b\u5ea6\u306b\u81ea\u52d5\u3067\u66f4\u65b0\u3055\u308c\u308b\u958b\u767a\u8005\u30e2\u30fc\u30c9\u7684\u306a":29,"\u305d\u306e\u3046\u3061":29,"\u305d\u308c\u306a\u3089":29,"\u305f\u3060\u9762\u5012\u306a\u306e\u3067":29,"\u3067\u30db\u30b9\u30c6\u30a3\u30f3\u30b0":29,"\u306e\u65b9\u304c\u8868\u73fe\u529b\u8c4a\u304b\u306a\u306e\u3067\u597d\u307e\u3057\u3044":29,"\u3082\u3057\u304f\u306f":29,"\u3082\u3063\u3068\u826f\u3044\u65b9\u6cd5\u304c\u3042\u308c\u3070\u6559\u3048\u3066\u307b\u3057\u3044":29,"\u3092\u4f7f\u3063\u3066\u81ea\u52d5\u5316\u3055\u305b\u308b\u4e88\u5b9a":29,"\u3092\u5b9f\u884c\u3057\u3066\u304b\u3089browser\u3067":29,"\u3092\u958b\u3051\u3070\u826f\u3044\u306f\u305a":29,"\u30d5\u30a1\u30a4\u30eb\u306f":29,"\u30d5\u30ec\u30fc\u30e0\u30ef\u30fc\u30af\u3092\u4f7f\u7528":29,"\u30ed\u30fc\u30ab\u30eb\u306e":29,"\u4ed6\u306b\u8cea\u554f\u304c\u3042\u308c\u3070github\u3084slack\u7d4c\u7531\u3067\u9023\u7d61\u304f\u3060\u3055\u3044":29,"\u53c2\u8003\u6587\u732e\u306e\u4f7f\u3044\u65b9\u304c\u5168\u822c\u7684\u306b\u8b0e\u3044":29,"\u5909\u63db\u5f8c\u306e\u30d5\u30a1\u30a4\u30eb\u3068\u753b\u50cf\u30d5\u30a1\u30a4\u30eb\u3092\u6b63\u3057\u3044\u30d5\u30a9\u30eb\u30c0\u306b\u5165\u308c\u308b":29,"\u65e5\u672c\u8a9e":0,"\u753b\u50cf\u30d5\u30a1\u30a4\u30eb\u306a\u3069\u306e\u9759\u7684\u306a\u3082\u306e\u306f":29,"\u9055\u3044\u304c\u3088\u304f\u308f\u304b\u3089\u3093":29,"boolean":[0,15],"build\u306e\u4e2d\u8eab\u3068read":29,"case":[1,12,24],"class":[1,8,10,12,14,15,17,20,22,24,26],"cos2\u03b8":[],"default":[14,15,20,29],"docs\u306e\u4e2d\u8eab\u304c\u5b8c\u5168\u4e00\u81f4\u3059\u308b\u308f\u3051\u3067\u306f\u306a\u3055\u305d\u3046":29,"float":15,"function":[1,3,4,5,8,9,10,15,23],"import":[1,3,5,6,7,8,10,11,12],"int":[14,15],"ipynb\u306fnbconvert\u3067":29,"long":1,"md\u306b\u5909\u63db\u3057\u3066\u304b\u3089":29,"new":[0,5],"return":[0,1,3,5,6,7,8,10,14,15],"short":26,"static\u306b\u5165\u308c\u308b":29,"true":[0,1,3,5,6,7,8,10,14,15],"try":0,"var":[6,7],"while":[8,10],Are:22,For:[1,3,22],Its:28,One:1,PCs:23,That:[],The:[0,1,3,8,10,14,17,22,23,24,28],Then:[],There:[24,26,29],These:3,Use:1,Uses:1,Using:[3,29],With:[8,10],__doc__:[8,10],__main__:[5,8,10],__name__:[5,8,10],_build:29,_static:[6,7],abd:17,abov:1,accuraci:[1,3],accuracy_scor:[1,3,11,12],across:14,add:[17,22,24],add_artist:[8,10],add_subplot:1,admin:0,advanc:[],advantag:[1,3],after:20,agg:[5,6,7,8],agreement:0,aim:23,air:12,akinari:28,algorithm:[1,3],all:[0,4,8,10,12,24,29],allow:[20,26],alpha:[3,6,7],alreadi:[1,3],also:[1,12,15,17,20,22,23,24,29],although:1,alwai:12,analysi:[3,4,6,7,9,24,27,28],angl:[1,3,6,7,8,10,15,22,24,26],ani:[0,6,7,12,14,17],anim:[6,7],anoth:3,anti:3,api:[1,3,11,12],append:3,appli:[0,3,17,22,24],applic:[22,28,29],apply_along_axi:7,approach:[20,26],appropri:20,approx:1,approxim:24,arang:[3,5],arcco:3,arctan:[8,10],area:[8,10],argv:5,around:26,arr:7,arrai:[0,1,6,7,8,10,11,12,14,15],artifici:12,artist:5,ascend:1,ask:0,assert:3,assign:[1,3],associ:0,assum:[1,3,22,24],astyp:14,atom:[5,6,7,8],atsuto:28,attempt:1,auto:1,avail:1,averag:0,avg:1,avoid:1,ax1:[3,6,7],ax2:[3,6,7],axes:[3,5,23],axhlin:[6,7],axi:[3,6,7,8,10],axs:3,axvlin:[6,7],backend:[5,6,7,8],bad:12,bai:17,barplot:1,base:[1,12,13,14,16,20,24,26,28],base_class:[12,13,16],baseestim:14,basi:[0,15,22,24],bbox:[8,10],becaus:[3,29],becom:[3,20,23,29],been:26,befor:[20,29],begin:[20,23],beginn:2,behavior:29,being:23,belong:15,below:[0,1,3,4,11,12,20,23],benefit:1,better:[1,3],between:[1,3,5,15,17,22,24,26],beween:15,biggest:15,black:[6,7,8,10],blit:[6,7],blue:[3,6,7,8,10],book:0,bool:[14,15],boston:28,both:[1,3],boundari:[1,3,8,10],break_ti:1,brows:29,build:[3,29],bulki:29,c_i:17,cache_s:1,calc:15,calc_basis_vector:15,calc_eer:15,calc_er:15,calcualt:20,calcul:[1,6,7,12,15,22,24],call:[1,5],calucl:1,can:[1,3,6,7,15,17,20,22,23,24,29],cannon:15,cannot:[5,6,7,8],canon:[15,17,22,26],canonical_angl:15,canonical_angle_matrix:15,canonical_angle_matrix_f:15,caption:[],captur:[3,20,23],caution:12,caveat:[1,29],cell:29,center:[4,6,9,12,20],cha12:[22,28],chang:[3,29],charact:28,chatelin2012eigenvalu:[],chatelin:28,check:[12,24],check_random_st:1,choic:1,choos:[1,3],cite:[],clafic:26,class_weight:1,classes_:1,classif:[1,12,17,20,22,24,26],classifi:[3,12,17,20,22,24,26],classification_report:1,classifiermixin:14,clear:29,clf:[1,3],click:[5,6,7,8,10],cm_bright:3,cmap:[1,3,8,10],cmsm:[12,13,16,24,26],cnn:28,code:[1,3,4,5,6,7,8,10,11,12,20,23,29],coef0:1,coincid:[],collabor:29,collect:1,collespond:15,color:[3,6,7,8,10],colormap:[8,10],column:1,com:[3,11,12],command:[11,12],commit:29,common:1,commonli:22,compar:1,comparison:[12,26],complet:29,compon:[0,3,6,7,14,17,24,27],comput:[1,12,26,28],computervisionlaboratori:[11,12],concept:12,conclus:[20,23],concret:[],condit:[20,23],conduct:[1,3],confer:28,confus:1,consid:3,consist:[],constant:[1,6,7],constrain:[14,26,27,28],constrainedmsm:14,constrainedsmbas:14,construct:[1,20],contact:12,contain:[1,12,14,17,29],content:[12,16,29],contour:[8,10],contourf:3,contribut:12,control:29,convert:[8,10],copi:[11,12],correl:24,correspond:[1,15,26],cos:[3,6,7,22,24],cos_sim:3,cosin:12,could:3,count:14,counter:1,cov:[6,7,8,10],covari:[3,4,6,7,9,20,23],covariance_:[8,10],creat:[3,6,7,29],credit:12,criteria:24,cross:1,cross_similar:15,crucial:1,current:[5,6,7,8],custom:[1,29],cvlab:[5,6,7,8,12],cvlab_toolbox:[5,6,7,8,11,29],cvt:[1,3,11,12],d_i:17,d_p:22,d_q:22,dark:[1,6,7,8,10],data:[0,1,3,8,10,11,12,14,15,17,20,23,24],data_typ:15,datafram:1,dataset:[0,6,7,8,10,12,20,23,24],dataset_cov:[8,10],dataset_fixed_cov:[6,7,8,10],dct:3,decemb:28,decid:22,decis:[1,3,8,10],decision_function_shap:1,decomposit:[6,7],deep:[0,14],def:[0,1,3,5,6,7,8,10],defin:[1,17,20,22,24],degre:[1,8,10],demo:12,demonstr:3,depend:1,depict:23,depth:12,deriv:[20,23],descend:15,describ:26,descript:0,detail:24,detect:1,determin:[3,17],dev:1,develop:26,deviat:[1,8,10],dic:3,dict:14,dictionari:1,differ:[1,3,8,10,28],differec:17,difficult:3,digit:1,dim:[6,7,8,10,11,12],dimens:[0,1,3,15,17,22],dimension:[1,3,17,22,23,26],diment:14,direct:[3,6,7,20,23],directli:[1,20],disadvantag:1,discrimin:[3,4,9,17,20,23],discriminant_analysi:[3,8,10],disp:1,displai:[1,8,10],dist:3,distanc:[1,3,15,26],distinct:1,distribut:[0,1,3,20,23,24,26],doc:[1,6,7,29],docstr:0,doe:[1,3,17],doi:28,dot:[6,7,8,10],doubl:[8,10],download:[4,5,6,7,8,10],dpi:5,dual:15,dual_vector:15,each:[0,1,5,8,10,12,14,15,17,22,24,26],easi:[3,11,12],easili:[1,3,22],edgecolor:[3,8,10],edit:[28,29],eer:15,effect:[1,17,26],effici:[1,29],effort:12,eig:[6,7],eig_val:[3,6,7],eig_vec:[3,6,7],eigen:[3,6,7,15,20,23],eigen_vec:3,eigenbasi:15,eigendecomposit:17,eigenvalu:[3,15,28],eigenvector:[6,7,12,15,17,20,23,24],eigh:[3,8,10],either:24,element:15,eleventh:28,elif:[8,10],ell:[8,10],ellips:[8,10],ellipsoid:[4,9],els:[3,5],empower:12,engin:12,english:0,enumer:[8,10],eps:15,equal:[3,15],equat:24,error:15,especi:3,essenc:3,estim:[1,14],euclidean:20,evalu:[12,13,16,28],exampl:[0,2,3,4,5,6,7,8,10,11,12,15,22],examples_script:9,examples_scripts_jupyt:4,examples_scripts_python:4,exaplan:20,except:29,exectut:1,execut:[1,9,29],exhaust:[1,3],exist:17,exp:[15,20],expens:1,experiment:1,explain:[20,23],explained_vari:0,explan:[],extend:[20,26],extens:[1,17,22,24,26],extract:17,extrem:29,face:28,facecolor:[8,10],fact:[3,24],fair:1,fals:[0,1,3,6,7,14,15],familiar:[20,23],farg:[6,7],fast_predict_proba:14,faster:3,faster_mod:[1,3,14],featur:[1,3,17,20,28],feel:12,fetch:1,fetch_openml:1,ff0000:3,fig:[1,3,5,6,7,17,22,24],fig_index:[8,10],figsiz:[1,6,7,8,10],figur:[1,5,6,7,8,10],figure_:1,file:[5,9],fill:[8,10],filterwarn:[1,3],find:[20,23],first:[3,23],fisher:17,fit:[1,3,8,10,11,12,14,23],five:1,fix:[8,10],flow:[],fm15:[17,28],focu:[20,23],fold:1,follow:[0,1,3,17,20,23,24,26],fontsiz:[8,10],forev:5,form:14,format:[1,5],format_input:[1,3],formul:[20,23],formula:[20,23],found:[1,20,23],fps:[6,7],frac:[20,22,23,24],frame:[5,6,7],framework:26,fran:28,free:12,from:[0,1,2,5,6,7,8,10,11,12,15,17,22,24,28,29],fuk14:[24,26,28],fukui2005fac:[],fukui2014:[],fukui:[12,28],full:[5,6,7,8,10],funcanim:[5,6,7],fundament:[3,24],further:[20,23],furthermor:[17,26],futur:1,fy05:[17,26,28],g_ratio:1,gait:28,galleri:[5,6,7,8,10,12,20,23],gamma:1,gaussian:[6,7,8,10],genchi:28,gener:[0,1,3,4,5,6,7,8,10,20,22,24,26,28,29],geomspac:[6,7],get:[3,11,12,14],get_dpi:5,get_param:14,get_size_inch:5,ggplot:3,gif:[5,6,7],git:[11,12,29],github:[11,12],give:3,given:[3,14,15,20],glanc:12,global:[6,7],goal:[20,23],going:3,golden:[1,6,7],good:12,goodfellow:0,graduat:12,grai:1,grammian:15,greater:1,greatli:20,green:[3,8,10],grei:[8,10],greyscal:1,ground:3,guarante:3,gui:[5,6,7,8],guid:0,hand:1,handl:20,hard:29,has:[1,8,10,26],hat:[],have:[3,5,14,22,26],head:1,help:12,here:[0,1,3,5,6,7,8,10,11,12,20,23,24],high:[1,3,26],higher:[3,15],highest:[17,22,24],highli:[1,29],hilbert:20,hiroshi:28,home:[5,6,7,8],host:12,how:[1,3,6,7,22,29],howev:[20,29],hstack:[8,10],html:29,http:[3,11,12,20,28,29],hue:1,human:29,hungri:3,hyper:20,hyperparamet:1,ian:0,ichi:28,ideal:17,ident:24,ieee:28,igm74:[24,26,28],ignor:[1,3],iijima1974theori:[],iijima:[26,28],iizuka:12,imag:[1,3,22,26,29],imagemagick:[5,6,7],implement:[1,20,26],implementaion:1,improv:12,imshow:1,inbal:1,inch:5,includ:[1,17,22],increas:17,independ:[15,26],index:[1,12,29],individu:12,infom:1,inform:[12,28],informat:12,init:[6,7],initi:5,inner:[15,20],input:[0,1,3,11,12,14,15,17,20,22,24,26],insert:[1,3],instanc:[1,14],instead:[17,22],integ:[0,1,14,15],intellig:[12,28],interest:[12,29],interfac:14,intern:[1,28],interpret:3,interv:[5,6,7],introduc:[3,17],introduct:[20,23],intuit:1,invers:3,invok:1,involv:12,ipynb:[5,6,7,8,10],is_norm:[6,7],isn:[5,6,7],item:[1,3,29],its:[8,10],japanes:26,javascript:29,join:12,jupyt:[4,5,6,7,8,10,29],jupyterlab:29,just:[12,17],kazuhiro:28,kcmsm:[12,13,16,20],keep:1,ken:28,kernel:[1,14,15,26,27],kernel_funct:[12,13,16],kernelcmsm:14,kernelcsmbas:14,kernelmsm:[11,12,14],kernelsmbas:14,kmsm:[12,13,16,20],kneighborsclassifi:1,knn:1,knnc:1,know:3,known:[23,26],kpca:20,l2_kernel:15,lab:[0,12],label:[1,3,5,6,7,11,12,14,15],labelcolor:3,laboratori:12,lambda:[15,17,20,23],larger:1,largest:[3,23],later:3,latest:12,latex:0,latter:14,lda:[8,10],lead:12,leaf_siz:1,learn:[0,1,8,10,11,12,20,26,28],leav:[20,23],lectur:28,left:[1,20],legend:3,legngth:3,len:[3,5,14],length:[1,3,24],less:1,let:[3,17,20,22,23],like:[1,12,14,15,29],limit:15,linalg:[3,6,7,8,10],line:[5,23],linear:[1,4,9,15,20],linear_kernel:15,lineardiscriminantanalysi:[3,8,10],linearli:[20,23],linearsegmentedcolormap:[8,10],lineplot:1,linestyl:[6,7],linewidth:[5,8,10],linspac:[3,8,10],list:[0,1,11,12,14,15],listedcolormap:3,lnq:[6,7],loc:[3,6,7],localhost:29,look:[3,12],loop:[1,5],lot:1,low:[3,17,22,24],lower:[3,15],m_j:[20,23],machin:[1,12,28,29],macro:1,made:[12,29],mae10:[20,23,28],maeda:28,mai:[1,17,22,24],mail:12,major:1,make:[0,1,3,23,29],make_circl:3,make_classif:3,make_moon:3,maki:28,manag:29,mani:[3,20,23,24,26,29],manual:29,map:14,marker:[8,10],markeredgecolor:[8,10],markers:[8,10],masashi:28,match:28,math:[],mathbb:[0,20,24],mathbf:[0,20,22,24],mathcal:[17,22],mathemat:[12,15],mathutil:15,matplotlib:[1,3,5,6,7,8,10],matric:[8,10,28],matrix:[0,1,3,6,7,8,10,14,15,17,20,23,24],max:3,max_:22,max_it:1,max_scor:3,max_square_singular_valu:15,maxdepth:[],maxim:[17,20,23],maximis:[6,7],mayb:1,mean:[1,3,4,6,8,9,10,15,20,23],mean_square_singular_valu:15,means_:[8,10],measur:26,media:29,melt:1,member:0,memori:1,mention:29,menu:29,merg:0,mesh:3,meshgrid:[3,8,10],method:[2,12,14,20,27,28],metric:[1,3,11,12,15],metric_param:1,min:3,minim:[4,9],minimum:[1,22,24,26],minkowski:1,minut:[5,6,7,8,10],mnist:[2,12],mnist_784:1,model:[1,3,11,12,13,16],model_select:[1,3,11,12],modul:[12,16],more:[1,3,24,26,29],mori:28,most:[1,3,26],motiv:12,mpl:[6,7,8,10],msm:[1,12,13,16,22,24,26],msminterfac:14,much:[0,1,3],multi:28,multipl:[17,26,28,29],multipli:[20,23],must:1,mutal:27,mutual:[14,17,22,26,28],mutualsubspacemethod:14,mva:28,n_class:[1,11,12,14],n_compon:3,n_dim:[1,14,15],n_dimens:15,n_element:15,n_featur:[0,3],n_gds_dim:14,n_input:15,n_job:1,n_neighbor:1,n_redund:3,n_ref:15,n_sampl:[0,1,3,14,15],n_samples_i:15,n_samples_x:15,n_set:15,n_set_i:15,n_set_x:15,n_subdim:[1,3,11,12,14,15],n_subdim_i:15,n_subdim_x:15,n_subdims_i:15,n_subdims_j:15,n_test:[11,12],n_train:[11,12],n_vector:15,n_vector_set:14,name:[1,14,20,26],nan:3,naoya:28,natur:26,nbsphinx:29,ncol:[3,6,7],ndarrai:[0,15],nearest:1,need:[1,5],neighbor:1,neq:22,nest:14,newli:3,next:[1,3],nishiyama:28,nois:1,nomralis:7,non:[1,5,6,7,8,20],none:[1,14,15],norm:[3,6,7,8,10],normal:[0,1,3,5,6,7,8,10,14,24],notat:12,note:[1,5,28],notebook:[4,5,6,7,8,10],now:3,nrow:[6,7],num:[6,7],number:[0,1,6,7,14,15,17],numpi:[0,1,3,5,6,7,8,10,11,12],numpydoc:0,nyf05:[17,28],object:14,obscur:29,obtain:[7,17,20,22,23,24],obviou:24,obvious:3,offici:12,often:24,ois:28,omsm:24,one:[1,12,26,29],ones:[3,8,10],onli:[1,3,17],onto:[3,17,23],openml:1,oper:20,optim:[1,3],option:15,order:15,org:[20,28],origin:[6,7,20,22],orthogan:17,orthogon:[17,23,24],orthonorm:[22,24],osamu:28,other:[3,24,26,29],otherhand:3,our:[3,12,20,23],out:[6,7,8,24],outlier:1,output:[20,29],over:[1,5],ovr:1,own:[8,10],packag:[3,12,16],page:[1,12,28,29],pairwis:15,panda:1,param:[1,14],param_nam:14,paramet:[0,1,14,15,20],pardir:[1,3],partial:[20,23],pass:1,past:[11,12],patch:[8,10],path:[1,3],pattern:[1,3,24,26,28],pca:[0,2,4,9,12,15,17,22,24],pcolormesh:[8,10],peform:12,pep8:0,per:1,perform:[1,3,6,7,17,20,26],perp:22,persist:[5,6,7],perspect:3,phase:[],phi:[6,7,17,20,23],phi_1:17,phi_d:17,phi_i:[17,24],pip:[11,12],pipelin:14,plane:[3,20],pleas:12,plot:[3,6,7,8,10,23,29],plot_ani:5,plot_confusion_matrix:1,plot_data:[8,10],plot_decision_boundari:3,plot_ellips:[8,10],plot_lda_cov:[8,10],plot_pca:[6,9],plot_qda_cov:[8,10],plot_stat:1,plot_subspace_pca:[7,9],plot_test:[8,9],plt:[1,3,5,6,7,8,10],png:3,point:[1,3,20,23,26],popular:1,posit:[8,10],possibl:[0,1,14,20],power:[3,26],practic:22,pre:29,precis:1,pred:[11,12,14],predefin:[6,7],predict:[1,8,10,11,12,14],predict_proba:[3,8,10,14],prepar:12,prerequiset:[],prerequisit:27,pretti:[11,12],previou:3,princip:[3,20,23,24],principl:[1,6,7,27],print:[1,3,5,8,10,11,12],proba2class:14,proba:14,probabl:[1,14],problem:3,procedur:[1,3],process:[],product:[15,20],prof:12,program:12,proj1:3,proj2:3,proj_vari:[6,7],project:[3,17,23,24],properti:14,provid:[1,3,5],pull:0,put:3,pyplot:[1,3,5,6,7,8,10],python:[0,4,5,6,7,8,10,29],q_i:24,qda:[8,10],qdf:[20,23],quadract:1,quadrat:[4,9,20,23],quadraticdiscriminantanalysi:[8,10],queri:[1,5],question:12,quiver:[3,6,7],rad2deg:3,rad:[6,7],rand:[11,12],randint:[1,11,12],randn:[6,7,8,10],random:[3,5,6,7,8,10,11,12],random_st:1,randomizedsearchcv:1,rang:[1,3,6,7,11,12,14],rate:15,ratio:17,ravel:[3,8,10],rbf:[1,15,20],rbf_kernel:15,rdbu:3,read:29,real:0,reason:29,recal:1,recognit:[26,28],recommend:29,red:[3,6,7,8,10],red_blue_class:[8,10],redrawn:5,redrawnstart_deg:[6,7],reduc:20,ref:15,refer:[3,12,15,17,26],reflect:22,register_cmap:[8,10],regress:1,regular:1,reject:[3,17,22,24],relat:3,remain:22,remeb:1,remov:29,replac:[20,23,26],replesent:15,repo:12,report:1,repositori:[12,29],repres:[1,3,15,20,23],represent:15,reproduc:20,reproducing_kernel_hilbert_spac:20,request:0,research:[12,26,28],reshap:[1,3,8,10],respect:26,restructur:29,result:[3,12,17],resus:3,return_eigv:15,return_x_i:1,revers:[6,7],review:0,revis:[1,28],rewritten:[20,23],rich:29,right:20,rigouru:24,robot:28,row:15,rst:29,rtd:12,rule:12,run:[1,5,6,7,8,10,11,12],sakai:28,same:[3,6,7,8,10,20,23,24],sampl:[1,6,7,8,10,14,20,23],satisfi:15,satosi:28,save:[5,6,7],scalar:0,scale:[1,3,6,7],scale_unit:[3,6,7],scatter:[3,5,6,7,8,10,23],school:12,scienc:28,scikit:[1,11,12,20],scipi:[1,6,7,8,10],score:[1,3],screen:5,script:[5,6,7,8,10],seaborn:[1,3,6,7],search:[1,3,12],second:[3,5,6,7,8,10,23],section:[1,3],see:[1,3,12,17,20,23,26,29],seed:[6,7,8,10],select:[12,28,29],self:14,sens:23,separ:[5,20],serv:12,set:[0,1,3,11,12,14,15,20,22,23,24,26],set_alpha:[8,10],set_aspect:3,set_clip_box:[8,10],set_data:[6,7],set_param:14,set_tight_layout:5,set_titl:[3,6,7],set_uvc:[6,7],set_xlabel:[3,5],set_xlim:[3,6,7],set_xtick:[8,10],set_ydata:5,set_ylabel:3,set_ylim:[3,6,7],set_ytick:[8,10],sever:[1,26],shape:[0,3,8,10,14,15],should:[3,11,12,24,29],show:[1,3,5,6,7,8,10],shown:[20,23],shrink:1,siam:28,sigma:[11,12,14,15,20],similar:[12,15,17,22,24,26],simpl:[1,14,15],simplehttpserv:29,simpli:1,simplic:[1,3],sin:[3,6,7],sinc:[1,24],singl:24,singular:15,size:[3,5,6,7],sklearn:[1,3,8,10,11,12,14,15],slight:1,smallest:22,smbase:14,smc:[1,3],sns:[1,3],sogi:28,solv:[6,7,20,23],solver:[8,10],some:0,sort:[3,6,7,15],sort_valu:1,sound:29,sourc:[4,5,6,7,8,10,17,20,22,23,29],sp_randint:1,space:[1,3,15,17,20,23,26],specifi:1,speed:1,sphinx:[4,5,6,7,8,10,29],splot:[8,10],springer:28,sqrt:15,squar:15,ssf19:[22,28],stabl:12,stackoverflow:3,standard:[0,8,10],start:[1,3,12,26],stat:1,std:1,step:[1,3,6,7,17],still:1,store:[1,3],store_covari:[8,10],stori:[20,23],string:[14,15],structur:[3,22],student:12,stuff:12,style:[1,3,6,7,12],submodul:[12,13,16],subobject:14,subpackag:[12,16],subplot:[3,5,6,7,8,10],subplots_adjust:[8,10],subset:1,subspac:[2,4,9,12,14,15,20,27,28],subspace_bas:15,subspace_pca:7,subspacemethod:[1,3,14],substack:22,suit:1,sum:[17,20,22,23,24],sum_:[20,23],supervis:[0,1],support:[12,20],suppress:1,suptitl:[1,6,7,8,10],survei:26,svc:1,svd:[8,10,22],svm:1,symbol:[0,1,3],symmetr:3,symposium:28,sys:[1,3,5],system:[12,28],taizo:28,take:[1,3],taken:0,target:[0,1,6,7,20,23],target_phi:[6,7],target_rad:[6,7],task:26,techniqu:24,tediou:29,tensor:0,term:[1,24],test:[0,1,10],test_n_subdim:14,test_siz:1,test_x:3,testset:1,text:29,than:1,thei:[3,22,24],them:26,theori:28,therefor:[3,20,29],thershold:[17,22,24],thesi:[5,6,7,8],theta:[3,24],theta_1:22,theta_i:22,thi:[1,3,5,6,7,8,10,12,14,15,17,20,22,23,24,26,29],think:[1,24],those:29,threashold:15,thresh:15,threshold:[],tick_param:3,tie:3,tight:[8,10],tight_layout:[8,10],tild:[17,22],time:[1,5,6,7,8,10,20],timeit:1,timestep:5,tip:3,titl:[1,8,10],toctre:[],toi:3,tol:1,too:3,toolbox:12,top:[8,10,15,17,20,24],total:[5,6,7,8,9,10],tp0:[8,10],tp1:[8,10],tpami:28,tqdm:1,track:1,train:[0,12,14,17,24],train_siz:1,train_test_split:[1,3,11,12],trainset:1,transact:28,transform:20,treat:[0,15],truth:3,tsukuba:12,tune:1,tupl:5,tutori:[1,3,12,24],twinx:3,two:[17,22,24,26],type:[1,14,15,29],typic:[3,26],u_1:22,u_i:[17,22],u_j:22,uncorrel:23,undersir:17,understand:3,uniform:1,uniqu:[1,3,14,22],unit:[12,20,23],unit_vector_from_rad:[6,7],univers:[12,20],unknown:[20,23],unnecessari:17,unrealist:20,updat:[5,6,7,14],update_quiv:[6,7],update_scatt:[6,7],url:28,use:[1,6,7,11,12,20,22],used:[0,1,3,17,20,22,23,29],useful:[1,17,29],userwarn:[5,6,7,8],uses:3,using:[1,3,5,6,7,8,14,15,17,20,22,24,28],usual:[17,20,23],util:[1,12,13,16],v_i:[17,22],v_j:22,val:[1,6,7],valid:1,valu:[1,14,15,22],vari:[8,10],variabl:[1,14,15,28],varianc:[0,3,6,7,20,23],variat:17,variou:12,vec:[6,7],vector:[0,3,7,11,12,14,15,17,20,22,23,24,26],verbos:1,veri:[3,29],versatil:1,version:1,via:12,viewpoint:28,vision:[12,26,28],visual:3,vote:1,wai:3,walkthrough:[12,24],warn:[1,3],wat67:[24,26,28],watanab:[26,28],watanabe1967evalu:[],web:12,websit:1,weight:1,well:[3,14,26],what:[3,15],when:[0,1,3,5,17,22],where:[1,3,15,17,22,24,26],which:[1,3,5,6,7,8,12,17,23,24],white:[8,10],whiten:0,whole:[],why:3,wiki:20,wikipedia:20,within:1,without:[4,9],won:1,work:[1,3,11,12,14,29],would:12,wrapper:15,write:0,writer:[5,6,7],written:1,x0_fp:[8,10],x0_tp:[8,10],x1_fp:[8,10],x1_tp:[8,10],x_a:[20,23],x_data:[6,7],x_i:[20,23],x_max:[3,8,10],x_min:[3,8,10],x_test:[1,11,12],x_train:[1,11,12],xix:[],xlabel:5,xlim:[3,8,10],y_data:[6,7],y_gt:3,y_max:[3,8,10],y_min:[3,8,10],y_pred:[1,3,8,10],y_test:[1,11,12],y_train:[1,11,12],yamaguchi:28,yellow:[8,10],ylabel:[8,10],ylim:[3,8,10],you:[3,12,17,22,24,29],your:[3,11,12,29],zero:[3,8,10,20,23],zip:[4,6,7],zorder:[8,10]},titles:["Coding styles","MNIST example with Subspace Method","Tutorials","From PCA to the Subspace Method","Gallery","Just testing how an animated plot turns out","PCA by minimizing the Quadratic Discriminant Function","Subspace PCA (PCA without mean centering)","Linear and Quadratic Discriminant Analysis with covariance ellipsoid","Computation times","Linear and Quadratic Discriminant Analysis with covariance ellipsoid","Installation","Welcome to cvlab_toolbox\u2019s documentation!","cvt package","cvt.models package","cvt.utils package","cvt","Constrained Subspace Method","&lt;no title&gt;","Kernel MSM / Kernel CMSM","Kernel Principle Component Analysis","&lt;no title&gt;","Mutal Subspace Method","Principle Component Analysis","Subspace Method","&lt;no title&gt;","Subspace Methods at a Glance","Concepts Walkthrough","References","\u958b\u767a\u8005\u30e1\u30e2"],titleterms:{"\u3092\u4f7f\u3046\u3068\u304d\u306e\u8af8\u6ce8\u610f":29,"\u30ed\u30fc\u30ab\u30eb\u3067\u5b9f\u884c\u3059\u308b\u65b9\u6cd5":29,"\u57fa\u672c\u7684\u306a\u3053\u3068":29,"\u8b0e":29,"\u958b\u767a\u8005\u30e1\u30e2":29,"case":3,"class":3,"function":[6,20],GDS:17,analysi:[8,10,20,23,26],anim:5,bad:3,base:15,base_class:14,basi:20,calcul:[3,17,20,23],center:7,classif:3,classifi:1,cmsm:[14,17,19],code:0,comparison:1,compon:[20,23],comput:9,concept:27,constrain:17,content:[13,14,15],contribut:0,cosin:3,covari:[8,10],cvlab_toolbox:12,cvt:[13,14,15,16],dataset:[1,3],depth:1,differ:17,discrimin:[6,8,10],document:12,each:3,eigenvector:3,ellipsoid:[8,10],evalu:15,exampl:1,from:3,galleri:4,gaussian:20,gener:17,glanc:26,good:3,how:5,improv:1,indic:12,instal:[11,12],just:5,kcmsm:14,kernel:[19,20],kernel_funct:15,kmsm:14,learn:[17,22,24],linear:[8,10],mathemat:0,mean:7,method:[1,3,17,22,24,26],minim:6,mnist:1,model:14,modul:[13,14,15],msm:[14,17,19],mutal:22,notat:0,notebook:29,out:5,packag:[13,14,15],pca:[3,6,7,20,23],peform:3,phase:[17,22,24],plot:5,prepar:[1,3],principl:[20,23],quadrat:[6,8,10],radial:20,recognit:[17,22,24],refer:28,result:1,rule:0,select:3,similar:3,style:0,submodul:[14,15],subpackag:13,subspac:[1,3,7,17,22,24,26],summari:[17,20,22,24],support:1,tabl:12,test:5,theori:[20,22,24],time:9,train:[1,3],trick:20,turn:5,tutori:2,use:3,util:15,vector:1,walkthrough:27,welcom:12,without:7}})