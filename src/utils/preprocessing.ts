import Statistics from "./statistics";
import { memoize } from "./helpers";
export interface PreprocessingConfig {
  differenceOrder?: number;
  seasonalPeriod?: number;
  seasonalOrder?: number;
  stationarityThreshold?: number;
}

const difference = (data: number[], order: number = 1): number[] => {
  if (data.length === 0 || order < 0) return [];
  if (order === 0) return [...data];

  let result = [...data];

  for (let d = 0; d < order; d++) {
    if (result.length <= 1) break;

    const temp: number[] = [];
    for (let i = 1; i < result.length; i++) {
      temp.push(result[i] - result[i - 1]);
    }
    result = temp;
  }

  return result;
};

const seasonalDifference = (
  data: number[],
  seasonalPeriod: number,
  order: number = 1
): number[] => {
  if (data.length === 0 || order < 0 || seasonalPeriod <= 0) return [];
  if (order === 0) return [...data];

  let result = [...data];

  for (let d = 0; d < order; d++) {
    if (result.length <= seasonalPeriod) break;

    const temp: number[] = [];
    for (let i = seasonalPeriod; i < result.length; i++) {
      temp.push(result[i] - result[i - seasonalPeriod]);
    }
    result = temp;
  }

  return result;
};

const undifference = (
  originalData: number[],
  diffData: number[],
  order: number = 1
): number[] => {
  if (diffData.length === 0 || order < 0) return [];
  if (order === 0) return [...diffData];
  if (originalData.length < order) return [];

  let result = [...diffData];

  for (let d = 0; d < order; d++) {
    const temp: number[] = [];
    temp.push(originalData[d]);

    for (let i = 0; i < result.length; i++) {
      temp.push(temp[temp.length - 1] + result[i]);
    }

    result = temp.slice(1);
  }

  return result;
};

const checkStationarity = memoize((data: number[], threshold: number = 0.95): boolean => {
  if (data.length < 2) return true;

  // simple ADF test using autocorrelation
  const acf = Statistics.autocorrelation(data, 1);
  return Math.abs(acf) < threshold;
});

const createDifferencingPipeline = (orders: number[]) =>
  (data: number[]): number[] =>
    orders.reduce((acc, order) => difference(acc, order), data);

const createSeasonalDifferencingPipeline = (
  seasonalPeriods: number[],
  orders: number[] = []
) => (data: number[]): number[] => {
  const orderArray = orders.length > 0 ? orders : new Array(seasonalPeriods.length).fill(1);

  return seasonalPeriods.reduce((acc, period, index) =>
    seasonalDifference(acc, period, orderArray[index] || 1), data
  );
};

const applyPreprocessing = (
  ...steps: Array<(data: number[]) => number[]>
) => (data: number[]): number[] =>
    steps.reduce((acc, step) => step(acc), data);

// find optimal d for differencing (useful for non-stationary data)
const findOptimalDifferencingOrder = (
  data: number[],
  maxOrder: number = 3,
  threshold: number = 0.95
): number => {
  for (let order = 0; order <= maxOrder; order++) {
    const diffData = difference(data, order);
    if (checkStationarity(diffData, threshold)) {
      return order;
    }
  }
  return maxOrder;
};

const findOptimalSeasonalPeriod = (
  data: number[],
  candidatePeriods: number[],
  threshold: number = 0.95
): number | null => {
  for (const period of candidatePeriods) {
    const seasonalDiffData = seasonalDifference(data, period);
    if (checkStationarity(seasonalDiffData, threshold)) {
      return period;
    }
  }
  return null;
};

const compose = <T>(...fns: Array<(arg: T) => T>) => (value: T): T =>
  fns.reduceRight((acc, fn) => fn(acc), value);


const createPreprocessor = (config: PreprocessingConfig) => {
  const {
    differenceOrder = 0,
    seasonalPeriod,
    seasonalOrder = 1,
    stationarityThreshold = 0.95
  } = config;

  return (data: number[]): number[] => {
    let result = [...data];

    if (differenceOrder > 0) {
      result = difference(result, differenceOrder);
    }

    if (seasonalPeriod && seasonalPeriod > 0) {
      result = seasonalDifference(result, seasonalPeriod, seasonalOrder);
    }

    if (!checkStationarity(result, stationarityThreshold)) {
      throw new Error('Processed data may not be stationary');
    }

    return result;
  };
};

export default {
  difference,
  seasonalDifference,
  undifference,
  checkStationarity,
  createDifferencingPipeline,
  createSeasonalDifferencingPipeline,
  applyPreprocessing,
  findOptimalDifferencingOrder,
  findOptimalSeasonalPeriod,
  createPreprocessor,
  compose
};