# ml_posture
This project is for my camp project.
It is for checking my posture when I sat in front of computer.
### Object
I want to detect my wrong posture using CNN. I choose Inception V3 model by Google. It is powerful model. 
I fed my posture picture on chair into Inception V3 model.



### How to use Inception v3
 There are many parameters in Inception V3 model. I think you have to fix these parameters(bottom of retrain code).
 These are environment valiable. 
 Add  --image_dir = 'path'
 
   >> parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    ) 
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument( 
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
>>    )
  
  
