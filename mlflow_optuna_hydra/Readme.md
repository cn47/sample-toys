# 実験管理コード
hydra+optuna+mlflowによる実験管理

## 本Repositoryのディレクトリ構成
```
.
├── Readme.md
├── data
│   ├── 01_raw              <- デモ用データ(kaggleのtitanicデータ: https://www.kaggle.com/competitions/titanic/data)
│   │   ├── test.csv
│   │   ├── train.csv
│   ├── hydra               <- hydraを仕込んだ.pyを実行すると自動的に作成され、各実行のたびログがユニークに実験ディレクトリが切られデータがたまる。場所や命名を変えたい場合はtrain.pyのmain函数前で書き換え可能
│   │   └── ....
│   └── mlflow
│       ├── _tmp
│       └── mlruns          <- mlflow trackingで参照する親ディレクトリ。mlflowを仕込んだ.pyを実行するとこれ以下に実験resourceがたまる。場所や命名を変えたい場合はtrain.pyのmain函数前で書き換え可能。変えた場合はmlflowサーバ立ち上げ時のoptionもそれに合わせて要変更
├── env
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── env_file.env
│   └── requirements.txt
└── src
    ├── config
    │   └── config.yaml     <- train.pyで参照するyaml。hydra経由で渡される。
    ├── logger.py
    ├── mlflow_writer.py     <- train.pyで参照するmlflowのカスタムクラス。このクラスで呼び出したインスタンスで実験resourceをmlflow管理化に保管していく
    ├── run_example.sh       <- train.pyの実行例
    ├── train.py             <- hydraを使ったLightGBMのハイパラ探索(optuna)＋ベストパラメータTrainコード。mlflowコード付き
    ├── train_hydra.py       <- hydraを使ったLightGBMのハイパラ探索(optuna)のみのコード
    └── utils.py
```


## つかいかた
### 環境構築について
実行環境はdocker(./envディレクトリ)で定義しています。  
docker環境を利用する人はdocker-compose.yml内の${UID}, ${GID}を自身の環境のもので上書きしてください  
(terminal上で`id`と打てば自分のUID/GIDが見れるのでそれで上書き)  

docker(docker-compose)環境がない人は、必要なpythonパッケージをローカルにいれてください。
``` sh
pip install -r ./env/requirements.txt
```


### mlflowサーバについて
dockerコンテナを立てたときにmlflowサーバを立ち上げてます。  
docker使わない場合はrequirements.txtをインストールした後以下のコマンドでサーバを立ち上げてください。
``` sh
mlflow server --backend-store-uri ./data/mlflow/mlruns -h 0.0.0.0 -p 5000
```
mlflowサーバはローカルに立てており、以下のURLに対してブラウザでアクセスします  
http://localhost:5000/


### hydraを使ったtrain.pyの実行方法
./src/run_example.shの中にterminalからの実行例を記載しています  
※ dockerを使わない場合は直接python3コマンドを打ってください。また、docker非利用者はtrain.py(train_hydra.py)のpj_dir変数を`./opt -> RepositoryのTop directory`に書き換えてください  


### 補足
#### LightGBMのwarningsについて
LightGBMのハイパラ探索のときにwarningsがたくさん出ますが煩わしいときはcross_validateを自分で実装してください。  
（fitのときにcallback渡せば消えるdeprecated warningsです）  
あるいはLightGBMTunerCVをつかってもよいとおもいます（ただしその場合複数のmetricsを引っ張り出せるかはわかりません）  
また厳密にはcross_validateの前にvalidデータを切ったほうが良い気もします（スコア的にあまり問題はないとはおもいますが）

#### GridSearch実行時について
GridSearchを実行するときはconfig.yamlを修正する必要があります。  
`hydra.sweeper`以下の要素と、下記の行をコメントアウトします。
```
defaults:
  - override hydra/sweeper: optuna
  - override hydra/sweeper/sampler: tpe
```
また、GridSearchをした場合、optuna実行時とちがってoptimization_results.yamlが生成されないので  
./src/train.pyのtrain_by_best_params()を実行しても最適ハイパラでモデルはつくられません（当デモの実装の問題）  
（そのとき作られるモデルは./src/config/config.yamlに記載されてるパラメータでfitしてます）  
このあたりはいずれなおそうかと..

