# Ridge Classifier Assessment
## 1. Intraday Data Fitting
### - Train Scope: 20240124, Test Scope: 20240125
### (1) Train Report
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-1.0</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.166667</td>
      <td>0.486653</td>
      <td>0.40000</td>
      <td>0.481928</td>
      <td>0.351107</td>
      <td>0.379793</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.007519</td>
      <td>0.979339</td>
      <td>0.01626</td>
      <td>0.481928</td>
      <td>0.334373</td>
      <td>0.481928</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.014388</td>
      <td>0.650206</td>
      <td>0.03125</td>
      <td>0.481928</td>
      <td>0.231948</td>
      <td>0.327525</td>
    </tr>
    <tr>
      <th>support</th>
      <td>133.000000</td>
      <td>242.000000</td>
      <td>123.00000</td>
      <td>0.481928</td>
      <td>498.000000</td>
      <td>498.000000</td>
    </tr>
  </tbody>
</table>
</div>

### (2) Test Report
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-1</th>
      <th>0</th>
      <th>1</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.0</td>
      <td>0.909134</td>
      <td>0.641509</td>
      <td>0.896722</td>
      <td>0.516881</td>
      <td>0.850572</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.0</td>
      <td>0.992226</td>
      <td>0.111475</td>
      <td>0.896722</td>
      <td>0.367900</td>
      <td>0.896722</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.0</td>
      <td>0.948865</td>
      <td>0.189944</td>
      <td>0.896722</td>
      <td>0.379603</td>
      <td>0.862006</td>
    </tr>
    <tr>
      <th>support</th>
      <td>829.0</td>
      <td>15307.000000</td>
      <td>915.000000</td>
      <td>0.896722</td>
      <td>17051.000000</td>
      <td>17051.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 2. 5 Days Fitting
### - Train Scope: 20240124-20240128, Test Scope: 20240129-20240130
### (1) Train Report
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: middle;
    }

    .dataframe thead th {
        text-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-1.0</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.282051</td>
      <td>0.365609</td>
      <td>0.356077</td>
      <td>0.359312</td>
      <td>0.334579</td>
      <td>0.334763</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.038128</td>
      <td>0.756477</td>
      <td>0.283531</td>
      <td>0.359312</td>
      <td>0.359379</td>
      <td>0.359312</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.067176</td>
      <td>0.492966</td>
      <td>0.315690</td>
      <td>0.359312</td>
      <td>0.291944</td>
      <td>0.292337</td>
    </tr>
    <tr>
      <th>support</th>
      <td>577.000000</td>
      <td>579.000000</td>
      <td>589.000000</td>
      <td>0.359312</td>
      <td>1745.000000</td>
      <td>1745.000000</td>
    </tr>
  </tbody>
</table>
</div>


### (2) Test Report
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: middle;
    }

    .dataframe thead th {
        text-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-1.0</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.044791</td>
      <td>0.927926</td>
      <td>0.056746</td>
      <td>0.721858</td>
      <td>0.343154</td>
      <td>0.857541</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.034535</td>
      <td>0.770671</td>
      <td>0.283086</td>
      <td>0.721858</td>
      <td>0.362764</td>
      <td>0.721858</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.039000</td>
      <td>0.842019</td>
      <td>0.094540</td>
      <td>0.721858</td>
      <td>0.325186</td>
      <td>0.779865</td>
    </tr>
    <tr>
      <th>support</th>
      <td>1332.000000</td>
      <td>31457.000000</td>
      <td>1413.000000</td>
      <td>0.721858</td>
      <td>34202.000000</td>
      <td>34202.000000</td>
    </tr>
  </tbody>
</table>
</div>


### (3) Validation
#### e.g.  y_true=-1, y_pred=1 -> "-1_to_1
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: middle;
    }

    .dataframe thead th {
        text-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>label</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0_to_0 </th>
      <td>24243</td>
    </tr>
    <tr>
      <th>1_to_0</th>
      <td>6283</td>
    </tr>
    <tr>
      <th>0_to_1</th>
      <td>0.963</td>
    </tr>
    <tr>
      <th>-1_to_0</th>
      <td>931</td>
    </tr>
    <tr>
      <th>0_to_-1</th>
      <td>920</td>
    </tr>
    <tr>
      <th>1_to_1</th>
      <td>400</td>
    </tr>
    <tr>
      <th>1_to_-1</th>
      <td>366</td>
    </tr>
    <tr>
      <th>-1_to_1</th>
      <td>50</td>
    </tr>
        <tr>
      <th>-1_to_-1</th>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>