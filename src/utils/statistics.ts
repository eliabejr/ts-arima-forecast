import { memoize } from "./helpers";

const mean = (data: number[]): number => {
  if (data.length === 0) return 0;
  return data.reduce((sum, val) => sum + val, 0) / data.length;
};

const variance = (data: number[]): number => {
  if (data.length <= 1) return 0;
  const meanValue = mean(data);
  return data.reduce((sum, val) => sum + Math.pow(val - meanValue, 2), 0) / (data.length - 1);
};

const standardDeviation = (data: number[]): number => {
  return Math.sqrt(variance(data));
};

const autocorrelation = memoize((data: number[], lag: number): number => {
  const n = data.length;
  if (n <= lag) return 0;

  const meanValue = mean(data);

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n - lag; i++) {
    numerator += (data[i] - meanValue) * (data[i + lag] - meanValue);
  }

  for (let i = 0; i < n; i++) {
    denominator += Math.pow(data[i] - meanValue, 2);
  }

  return denominator === 0 ? 0 : numerator / denominator;
});

const solveLinearSystem = (matrix: number[][], rhs: number[]): number[] => {
  const n = matrix.length;
  if (n === 0) return [];

  const augmented = matrix.map((row, i) => [...row, rhs[i]]);

  // gaussian elimination
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
        maxRow = k;
      }
    }

    // swap rows
    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

    if (Math.abs(augmented[i][i]) < 1e-10) {
      throw new Error('Matrix is singular or nearly singular');
    }

    // make all rows below this one 0 in current column
    for (let k = i + 1; k < n; k++) {
      const factor = augmented[k][i] / augmented[i][i];
      for (let j = i; j <= n; j++) {
        augmented[k][j] -= factor * augmented[i][j];
      }
    }
  }

  // back substitution
  const solution: number[] = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    solution[i] = augmented[i][n];
    for (let j = i + 1; j < n; j++) {
      solution[i] -= augmented[i][j] * solution[j];
    }
    solution[i] /= augmented[i][i];
  }

  return solution;
};

const partialAutocorrelation = memoize((data: number[], maxLag: number): number[] => {
  const n = data.length;
  if (n === 0 || maxLag < 0) return [];

  const pacf: number[] = [];

  // PACF[0] = 1 by definition
  pacf[0] = 1;

  if (maxLag === 0) return pacf;

  // PACF[1] = ACF[1]
  pacf[1] = autocorrelation(data, 1);

  // Calculate PACF using Yule-Walker equations
  for (let k = 2; k <= maxLag; k++) {
    const matrix: number[][] = [];
    const rhs: number[] = [];

    for (let i = 0; i < k; i++) {
      const row: number[] = [];
      for (let j = 0; j < k; j++) {
        row.push(autocorrelation(data, Math.abs(i - j)));
      }
      matrix.push(row);
      rhs.push(autocorrelation(data, i + 1));
    }

    try {
      const solution = solveLinearSystem(matrix, rhs);
      pacf[k] = solution[k - 1];
    } catch (error) {
      pacf[k] = 0;
    }
  }

  return pacf;
});

const akaike = (logLikelihood: number, numParams: number, n: number): number => {
  return 2 * numParams - 2 * logLikelihood;
};

const bayesian = (logLikelihood: number, numParams: number, n: number): number => {
  if (n <= 0) throw new Error('Sample size must be positive');
  return Math.log(n) * numParams - 2 * logLikelihood;
};

// utility function for composing statistical operations
const compose = <T>(...fns: Array<(arg: T) => T>) => (value: T): T =>
  fns.reduceRight((acc, fn) => fn(acc), value);

const createStatsPipeline = <T>(
  ...operations: Array<(data: number[]) => T>
) => (data: number[]): T[] => operations.map(op => op(data));

const leastSquares = (X: number[][], y: number[]): number[] => {
  const n = X.length;
  const p = X[0].length;

  // Calculate X'X
  const XtX: number[][] = [];
  for (let i = 0; i < p; i++) {
    XtX[i] = [];
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += X[k][i] * X[k][j];
      }
      XtX[i][j] = sum;
    }
  }

  // Calculate X'y
  const Xty: number[] = [];
  for (let i = 0; i < p; i++) {
    let sum = 0;
    for (let k = 0; k < n; k++) {
      sum += X[k][i] * y[k];
    }
    Xty[i] = sum;
  }

  return solveLinearSystem(XtX, Xty);
};

const getZScore = (confidenceLevel: number): number => {
  const p = 1 - (1 - confidenceLevel) / 2;

  // Abramowitz and Stegun formula 26.2.23 approximation
  const approximateInverseCDF = (p: number): number => {
    const plow = 0.02425;
    const phigh = 1 - plow;

    const c = [
      -0.007784894002430293,
      -0.3223964580411365,
      -2.400758277161838,
      -2.549732539343734,
      4.374664141464968,
      2.938163982698783,
    ];

    const d = [
      0.007784695709041462,
      0.3224671290700398,
      2.445134137142996,
      3.754408661907416,
    ];

    const a = [
      -39.69683028665376,
      220.9460984245205,
      -275.9285104469687,
      138.3577518672690,
      -30.66479806614716,
      2.506628277459239,
    ];

    const b = [
      -54.47609879822406,
      161.5858368580409,
      -155.6989798598866,
      66.80131188771972,
      -13.28068155288572,
    ];

    const rational = (
      x: number,
      num: number[],
      den: number[]
    ): number =>
      num.reduce((acc, coeff) => acc * x + coeff, 0) /
      (den.reduce((acc, coeff) => acc * x + coeff, 0) + 1);

    if (p < plow) {
      const q = Math.sqrt(-2 * Math.log(p));
      return -rational(q, c, d);
    }

    if (p > phigh) {
      const q = Math.sqrt(-2 * Math.log(1 - p));
      return rational(q, c, d);
    }

    const q = p - 0.5;
    const r = q * q;
    return rational(r, a, b) * q;
  };

  return approximateInverseCDF(p);
};


export default {
  mean,
  variance,
  standardDeviation,
  autocorrelation,
  partialAutocorrelation,
  akaike,
  bayesian,
  compose,
  createStatsPipeline,
  leastSquares,
  solveLinearSystem,
  getZScore
};
