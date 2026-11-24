/**
 * Recorded output of search.run.ts: for every case, every single change in the fix bench's menu and
 * the accuracy on fresh data it reached. Do not edit by hand; re-run the search. clinic.test.ts
 * re-trains a sample to check these are genuine.
 */
export const FIXES: Record<string, { label: string; fresh: number }[]> = {
 "flatline": [
  {
   "label": "step size 0.0003",
   "fresh": 0.8783
  },
  {
   "label": "step size 0.001",
   "fresh": 0.8833
  },
  {
   "label": "step size 0.003",
   "fresh": 0.9033
  },
  {
   "label": "step size 0.01",
   "fresh": 0.985
  },
  {
   "label": "step size 0.03",
   "fresh": 0.9833
  },
  {
   "label": "step size 0.1",
   "fresh": 0.98
  },
  {
   "label": "step size 0.3",
   "fresh": 0.9833
  },
  {
   "label": "step size 1",
   "fresh": 0.4583
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.98
  },
  {
   "label": "optimizer adam",
   "fresh": 0.9067
  },
  {
   "label": "activation relu",
   "fresh": 0.5
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.795
  },
  {
   "label": "hidden 4",
   "fresh": 0.5
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.7933
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.5
  },
  {
   "label": "batch size 1",
   "fresh": 0.86
  },
  {
   "label": "batch size 50",
   "fresh": 0.5
  },
  {
   "label": "500 steps",
   "fresh": 0.5
  },
  {
   "label": "8000 steps",
   "fresh": 0.5
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.5
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.9067
  },
  {
   "label": "stop shuffling",
   "fresh": 0.5
  }
 ],
 "tortoise": [
  {
   "label": "step size 0.001",
   "fresh": 0.8567
  },
  {
   "label": "step size 0.003",
   "fresh": 0.8783
  },
  {
   "label": "step size 0.01",
   "fresh": 0.8833
  },
  {
   "label": "step size 0.03",
   "fresh": 0.9017
  },
  {
   "label": "step size 0.1",
   "fresh": 0.985
  },
  {
   "label": "step size 0.3",
   "fresh": 0.985
  },
  {
   "label": "step size 1",
   "fresh": 0.985
  },
  {
   "label": "step size 3",
   "fresh": 0.98
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.8783
  },
  {
   "label": "optimizer adam",
   "fresh": 0.885
  },
  {
   "label": "activation relu",
   "fresh": 0.8567
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.455
  },
  {
   "label": "hidden 4",
   "fresh": 0.745
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.825
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.8
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.8717
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.8717
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.87
  },
  {
   "label": "batch size 1",
   "fresh": 0.8733
  },
  {
   "label": "batch size 50",
   "fresh": 0.8733
  },
  {
   "label": "500 steps",
   "fresh": 0.815
  },
  {
   "label": "8000 steps",
   "fresh": 0.855
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.865
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.8733
  },
  {
   "label": "stop shuffling",
   "fresh": 0.8733
  }
 ],
 "doa": [
  {
   "label": "step size 0.0003",
   "fresh": 0.5
  },
  {
   "label": "step size 0.001",
   "fresh": 0.5
  },
  {
   "label": "step size 0.003",
   "fresh": 0.5
  },
  {
   "label": "step size 0.03",
   "fresh": 0.5
  },
  {
   "label": "step size 0.1",
   "fresh": 0.5
  },
  {
   "label": "step size 0.3",
   "fresh": 0.5
  },
  {
   "label": "step size 1",
   "fresh": 0.5
  },
  {
   "label": "step size 3",
   "fresh": 0.5
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.5
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.5
  },
  {
   "label": "activation tanh",
   "fresh": 0.9817
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.8817
  },
  {
   "label": "hidden 4",
   "fresh": 0.5
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.5
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.5
  },
  {
   "label": "batch size 1",
   "fresh": 0.5
  },
  {
   "label": "batch size 50",
   "fresh": 0.5
  },
  {
   "label": "500 steps",
   "fresh": 0.5
  },
  {
   "label": "8000 steps",
   "fresh": 0.5
  },
  {
   "label": "standard initialisation",
   "fresh": 0.9833
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.5
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.5
  },
  {
   "label": "stop shuffling",
   "fresh": 0.5
  }
 ],
 "stiff": [
  {
   "label": "step size 0.0003",
   "fresh": 0.4533
  },
  {
   "label": "step size 0.001",
   "fresh": 0.4083
  },
  {
   "label": "step size 0.003",
   "fresh": 0.3717
  },
  {
   "label": "step size 0.03",
   "fresh": 0.4067
  },
  {
   "label": "step size 0.1",
   "fresh": 0.41
  },
  {
   "label": "step size 0.3",
   "fresh": 0.405
  },
  {
   "label": "step size 1",
   "fresh": 0.425
  },
  {
   "label": "step size 3",
   "fresh": 0.5
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.4067
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.3667
  },
  {
   "label": "activation relu",
   "fresh": 0.3683
  },
  {
   "label": "hidden 4",
   "fresh": 0.9467
  },
  {
   "label": "hidden 8 × 8",
   "fresh": 0.9667
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.9583
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.96
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.3683
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.3617
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.3283
  },
  {
   "label": "batch size 1",
   "fresh": 0.5333
  },
  {
   "label": "batch size 50",
   "fresh": 0.4417
  },
  {
   "label": "500 steps",
   "fresh": 0.485
  },
  {
   "label": "8000 steps",
   "fresh": 0.345
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.3383
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.3767
  },
  {
   "label": "stop shuffling",
   "fresh": 0.38
  }
 ],
 "memoriser": [
  {
   "label": "step size 0.0003",
   "fresh": 0.7483
  },
  {
   "label": "step size 0.001",
   "fresh": 0.7833
  },
  {
   "label": "step size 0.003",
   "fresh": 0.7783
  },
  {
   "label": "step size 0.03",
   "fresh": 0.7683
  },
  {
   "label": "step size 0.1",
   "fresh": 0.6217
  },
  {
   "label": "step size 0.3",
   "fresh": 0.7367
  },
  {
   "label": "step size 1",
   "fresh": 0.5
  },
  {
   "label": "step size 3",
   "fresh": 0.5
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.745
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.7767
  },
  {
   "label": "activation relu",
   "fresh": 0.8283
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.4583
  },
  {
   "label": "hidden 4",
   "fresh": 0.7383
  },
  {
   "label": "hidden 8 × 8",
   "fresh": 0.7583
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.7733
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.785
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.755
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.5
  },
  {
   "label": "batch size 1",
   "fresh": 0.6967
  },
  {
   "label": "batch size 50",
   "fresh": 0.7667
  },
  {
   "label": "500 steps",
   "fresh": 0.7617
  },
  {
   "label": "2000 steps",
   "fresh": 0.76
  },
  {
   "label": "8000 steps",
   "fresh": 0.7483
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.8633
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.7583
  },
  {
   "label": "stop shuffling",
   "fresh": 0.775
  }
 ],
 "too-good": [
  {
   "label": "step size 0.0003",
   "fresh": 0.7483
  },
  {
   "label": "step size 0.001",
   "fresh": 0.7833
  },
  {
   "label": "step size 0.003",
   "fresh": 0.7783
  },
  {
   "label": "step size 0.03",
   "fresh": 0.7683
  },
  {
   "label": "step size 0.1",
   "fresh": 0.6217
  },
  {
   "label": "step size 0.3",
   "fresh": 0.7367
  },
  {
   "label": "step size 1",
   "fresh": 0.5
  },
  {
   "label": "step size 3",
   "fresh": 0.5
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.745
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.7767
  },
  {
   "label": "activation relu",
   "fresh": 0.8283
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.4583
  },
  {
   "label": "hidden 4",
   "fresh": 0.7383
  },
  {
   "label": "hidden 8 × 8",
   "fresh": 0.7583
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.7733
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.785
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.755
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.5
  },
  {
   "label": "batch size 1",
   "fresh": 0.6967
  },
  {
   "label": "batch size 50",
   "fresh": 0.7667
  },
  {
   "label": "500 steps",
   "fresh": 0.7617
  },
  {
   "label": "2000 steps",
   "fresh": 0.76
  },
  {
   "label": "8000 steps",
   "fresh": 0.7483
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.8633
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.7583
  },
  {
   "label": "stop shuffling",
   "fresh": 0.775
  },
  {
   "label": "remove duplicates from the test set",
   "fresh": 0.7533
  }
 ],
 "random": [
  {
   "label": "step size 0.0003",
   "fresh": 0.5683
  },
  {
   "label": "step size 0.001",
   "fresh": 0.6217
  },
  {
   "label": "step size 0.003",
   "fresh": 0.5
  },
  {
   "label": "step size 0.03",
   "fresh": 0.5
  },
  {
   "label": "step size 0.1",
   "fresh": 0.5
  },
  {
   "label": "step size 0.3",
   "fresh": 0.5
  },
  {
   "label": "step size 1",
   "fresh": 0.5
  },
  {
   "label": "step size 3",
   "fresh": 0.5
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.5733
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.5433
  },
  {
   "label": "activation relu",
   "fresh": 0.54
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.6283
  },
  {
   "label": "hidden 4",
   "fresh": 0.43
  },
  {
   "label": "hidden 8 × 8",
   "fresh": 0.4717
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.5867
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.5
  },
  {
   "label": "batch size 1",
   "fresh": 0.5
  },
  {
   "label": "batch size 50",
   "fresh": 0.4383
  },
  {
   "label": "500 steps",
   "fresh": 0.4717
  },
  {
   "label": "2000 steps",
   "fresh": 0.5367
  },
  {
   "label": "8000 steps",
   "fresh": 0.5
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.1683
  },
  {
   "label": "fix the labels",
   "fresh": 0.985
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.3783
  },
  {
   "label": "stop shuffling",
   "fresh": 0.5217
  }
 ],
 "units": [
  {
   "label": "step size 0.0003",
   "fresh": 0.8433
  },
  {
   "label": "step size 0.001",
   "fresh": 0.6767
  },
  {
   "label": "step size 0.003",
   "fresh": 0.6683
  },
  {
   "label": "step size 0.03",
   "fresh": 0.6567
  },
  {
   "label": "step size 0.1",
   "fresh": 0.655
  },
  {
   "label": "step size 0.3",
   "fresh": 0.655
  },
  {
   "label": "step size 1",
   "fresh": 0.5
  },
  {
   "label": "step size 3",
   "fresh": 0.6483
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.6733
  },
  {
   "label": "optimizer adam",
   "fresh": 0.9667
  },
  {
   "label": "activation relu",
   "fresh": 0.5
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.6733
  },
  {
   "label": "hidden 4",
   "fresh": 0.655
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.6617
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.655
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.655
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.655
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.655
  },
  {
   "label": "batch size 1",
   "fresh": 0.655
  },
  {
   "label": "batch size 50",
   "fresh": 0.655
  },
  {
   "label": "500 steps",
   "fresh": 0.6683
  },
  {
   "label": "8000 steps",
   "fresh": 0.655
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.6583
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.985
  },
  {
   "label": "stop shuffling",
   "fresh": 0.655
  }
 ],
 "seesaw": [
  {
   "label": "step size 0.0003",
   "fresh": 0.8733
  },
  {
   "label": "step size 0.001",
   "fresh": 0.8517
  },
  {
   "label": "step size 0.003",
   "fresh": 0.865
  },
  {
   "label": "step size 0.01",
   "fresh": 0.865
  },
  {
   "label": "step size 0.03",
   "fresh": 0.815
  },
  {
   "label": "step size 0.1",
   "fresh": 0.755
  },
  {
   "label": "step size 1",
   "fresh": 0.9483
  },
  {
   "label": "step size 3",
   "fresh": 0.5
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.5
  },
  {
   "label": "optimizer adam",
   "fresh": 0.7467
  },
  {
   "label": "activation relu",
   "fresh": 0.97
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.8483
  },
  {
   "label": "hidden 4",
   "fresh": 0.72
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.5
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.5717
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.52
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.5
  },
  {
   "label": "batch size 1",
   "fresh": 0.6
  },
  {
   "label": "batch size 50",
   "fresh": 0.9817
  },
  {
   "label": "500 steps",
   "fresh": 0.6967
  },
  {
   "label": "8000 steps",
   "fresh": 0.9817
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.5
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.7433
  },
  {
   "label": "shuffle the training data",
   "fresh": 0.97
  }
 ],
 "quiet": [
  {
   "label": "step size 0.0003",
   "fresh": 0.9
  },
  {
   "label": "step size 0.001",
   "fresh": 0.9683
  },
  {
   "label": "step size 0.003",
   "fresh": 0.9817
  },
  {
   "label": "step size 0.01",
   "fresh": 0.9833
  },
  {
   "label": "step size 0.03",
   "fresh": 0.9817
  },
  {
   "label": "step size 0.1",
   "fresh": 0.985
  },
  {
   "label": "step size 0.3",
   "fresh": 0.92
  },
  {
   "label": "step size 3",
   "fresh": 0.5
  },
  {
   "label": "optimizer sgd",
   "fresh": 0.98
  },
  {
   "label": "optimizer momentum",
   "fresh": 0.5
  },
  {
   "label": "activation tanh",
   "fresh": 0.8817
  },
  {
   "label": "hidden no hidden layer",
   "fresh": 0.875
  },
  {
   "label": "hidden 4",
   "fresh": 0.5
  },
  {
   "label": "hidden 16 × 16",
   "fresh": 0.5
  },
  {
   "label": "hidden 16 × 16 × 16",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.001",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.01",
   "fresh": 0.5
  },
  {
   "label": "weight decay 0.1",
   "fresh": 0.5
  },
  {
   "label": "batch size 1",
   "fresh": 0.5
  },
  {
   "label": "batch size 50",
   "fresh": 0.4983
  },
  {
   "label": "500 steps",
   "fresh": 0.6433
  },
  {
   "label": "8000 steps",
   "fresh": 0.645
  },
  {
   "label": "collect 4× more data",
   "fresh": 0.5
  },
  {
   "label": "normalise the inputs",
   "fresh": 0.8683
  },
  {
   "label": "stop shuffling",
   "fresh": 0.5
  }
 ]
};
