# Install
- Note: you need to install python3.11 (use brew to install it)
  - in your shell rc, you can create an alias so pip points to your specific pip implementation 
- pip3 install --upgrade pip
- pip3 install tensorflow
- pip3 install pandas
- pip3 install gym
- pip3 install scikit-learn

# Run
- Run generate_data.py
- Run trainer.py
- Run tester.py
- Run retrainer.py
- Start using it via APIs

# Test API
- Run the command `fastapi dev main.py` in terminal
- Navigate to http://127.0.0.1:8000/docs to see available methods and test

# Resources
- [TensorFlow 2.0 Crash Course](https://www.youtube.com/watch?v=6g4O5UOH304&t=427s)
- [TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk)
- [Reinforcement Learning](https://colab.research.google.com/drive/1IlrlS3bB8t1Gd5Pogol4MIwUxlAjhWOQ#forceEdit=true&sandboxMode=true)
- [Creating a Reinforcement Learning Model with Tensorflow](https://medium.com/@aryanjha/creating-a-reinforcement-learning-model-with-tensorflow-39f97dfb0ee6)
- [Fastapi Documentation](https://fastapi.tiangolo.com/)