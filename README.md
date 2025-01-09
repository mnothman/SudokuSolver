python main.py --input data/inputs/sample_sudoku.jpg --output data/outputs/solved_sudoku.jpg

python src/ocr.py

python src/main.py --input data/inputs/sample_sudoku.jpg --output data/outputs/solved_sudoku.jpg

https://www.kaggle.com/datasets/bryanpark/sudoku


env

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt


----------------------------
SudokuSolver/
│
├── data/
│   ├── datasets/       
│   │   ├── sudoku.csv 
│   │   └── ...
│   ├── inputs/         
│   ├── outputs/        
│   └── ...
│
├── models/             
│   ├── sudoku_solver.keras  
│   └── ...
│
├── src/                 
│   ├── __init__.py      
│   ├── data_loader.py   
│   ├── model.py      
│   ├── train.py       
│   ├── solver.py       
│   └── ...
│
├── tests/               
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── ...
│
├── notebooks/          
│   ├── sudoku_training.ipynb
│   ├── sudoku_solver_demo.ipynb
│   └── ...
│
├── requirements.txt      
├── README.md            
└── main.py
