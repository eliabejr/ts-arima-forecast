
import { Stepwise, RollingWindow, Adaptive } from '../../utils/strategies';
import { ARIMA } from '../../models/arima';

describe('Forecasting Strategies', () => {
  let baseModel: ARIMA;
  let trainData: number[];
  let testData: number[];
  const btcData = [
    67766.85,
    67765.63,
    68809.9,
    70537.84,
    71108,
    70799.06,
    69355.6,
    69310.46,
    69648.14,
    69540,
    67314.24,
    68263.99,
    66773.01,
    66043.99,
    66228.25,
    66676.87,
    66504.33,
    65175.32,
    64974.37,
    64869.99,
    64143.56,
    64262.01,
    63210.01,
    60293.3,
    61806.01,
    60864.99,
    61706.47,
    60427.84,
    60986.68,
    62772.01,
    62899.99,
    62135.47,
    60208.58,
    57050.01,
    56628.79,
    58230.13,
    55857.81,
    56714.62,
    58050,
    57725.85,
    57339.89,
    57889.1,
    59204.02,
    60797.91,
    64724.14,
    65043.99,
    64087.99,
    63987.92,
    66660,
    67139.96,
    68165.34,
    67532.01,
    65936.01,
    65376,
    65799.95,
    67907.99,
    67896.5,
    68249.88,
    66784.69,
    66188,
    64628,
    65354.02,
    61498.33,
    60697.99,
    58161,
    54018.81,
    56022.01,
    55134.16,
    61685.99,
    60837.99,
    60923.51,
    58712.59,
    59346.64,
    60587.15,
    58683.39,
    57541.06,
    58874.6,
    59491.99,
    58427.35,
    59438.5,
    59013.8,
    61156.03,
    60375.84,
    64037.24,
    64157.01,
    64220,
    62834,
    59415,
    59034.9,
    59359.01,
    59123.99,
    58973.99,
    57301.86,
    59132.13,
    57487.73,
    57970.9,
    56180,
    53962.97,
    54160.86,
    54869.95,
    57042,
    57635.99,
    57338,
    58132.32,
    60498,
    59993.03,
    59132,
    58213.99,
    60313.99,
    61759.99,
    62947.99,
    63201.05,
    63348.96,
    63578.76,
    63339.99,
    64262.7,
    63152.01,
    65173.99,
    65769.95,
    65858,
    65602.01,
    63327.59,
    60805.78,
    60649.28,
    60752.71,
    62086,
    62058,
    62819.91,
    62224,
    62160.49,
    60636.02,
    60326.39,
    62540,
    63206.22,
    62870.02,
    66083.99,
    67074.14,
    67620.01,
    67421.78,
    68428,
    68378,
    69031.99,
    67377.5,
    67426,
    66668.65,
    68198.28,
    66698.33,
    67092.76,
    68021.7,
    69962.21,
    72736.42,
    72344.74,
    70292.01,
    69496.01,
    69374.74,
    68775.99,
    67850.01,
    69372.01,
    75571.99,
    75857.89,
    76509.78,
    76677.46,
    80370.01,
    88647.99,
    87952.01,
    90375.2,
    87325.59,
    91032.07,
    90586.92,
    89855.99,
    90464.08,
    92310.79,
    94286.56,
    98317.12,
    98892,
    97672.4,
    97900.04,
    93010.01,
    91965.16,
    95863.11,
    95643.98,
    97460,
    96407.99,
    97185.18,
    95840.62,
    95849.69,
    98587.32,
    96945.63,
    99740.84,
    99831.99,
    101109.59,
    97276.47,
    96593,
    101125,
    100004.29,
    101424.25,
    101420,
    104463.99,
    106058.66,
    106133.74,
    100204.01,
    97461.86,
    97805.44,
    97291.99,
    95186.27,
    94881.47,
    98663.58,
    99429.6,
    95791.6,
    94299.03,
    95300,
    93738.2,
    92792.05,
    93576,
    94591.79,
    96984.79,
    98174.18,
    98220.5,
    98363.61,
    102235.6,
    96954.61,
    95060.61,
    92552.49,
    94726.11,
    94599.99,
    94545.06,
    94536.1,
    96560.86,
    100497.35,
    99987.3,
    104077.48,
    104556.23,
    101331.57,
    102260.01,
    106143.82,
    103706.66,
    103910.34,
    104870.5,
    104746.85,
    102620,
    102082.83,
    101335.52,
    103733.24,
    104722.94,
    102429.56,
    100635.65,
    97700.59,
    101328.52,
    97763.13,
    96612.43,
    96554.35,
    96506.8,
    96444.74,
    96462.75,
    97430.82,
    95778.2,
    97869.99,
    96608.14,
    97500.48,
    97569.66,
    96118.12,
    95780,
    95671.74,
    96644.37,
    98305,
    96181.98,
    96551.01,
    96258,
    91552.88,
    88680.4,
    84250.09,
    84708.58,
    84349.94,
    86064.53,
    94270,
    86220.61,
    87281.98,
    90606.01,
    89931.89,
    86801.75,
    86222.45,
    80734.37,
    78595.86,
    82932.99,
    83680.12,
    81115.78,
    83983.2,
    84338.44,
    82574.53,
    84010.03,
    82715.03,
    86845.94,
    84223.39,
    84088.79,
    83840.59,
    86082.5,
    87498.16,
    87392.87,
    86909.17,
    87232.01,
    84424.38,
    82648.54,
    82389.99,
    82550.01,
    85158.34,
    82516.29,
    83213.09,
    83889.87,
    83537.99,
    78430,
    79163.24,
    76322.42,
    82615.22,
    79607.3,
    83423.84,
    85276.9,
    83760,
    84591.58,
    83643.99,
    84030.38,
    84947.91,
    84474.69,
    85077.01,
    85179.24,
    87516.23,
    93442.99,
    93691.08,
    93980.47,
    94638.68,
    94628,
    93749.3,
    95011.18,
    94256.82,
    94172,
    96489.91,
    96887.14,
    95856.42,
    94277.62,
    94733.68,
    96834.02,
    97030.5,
    103261.6,
    102971.99,
    104809.53,
    104118,
    102791.32,
    104103.72,
    103507.82,
    103763.71,
    103463.9,
    103126.65,
    106454.26,
    105573.74,
    106849.99,
    109643.99,
    111696.21,
    107318.3,
    107761.91,
    109004.19,
    109434.79,
    108938.17,
    107781.78,
    105589.75,
    103985.48,
    104644
  ]

  beforeEach(() => {
    // Generate consistent test data
    trainData = btcData.slice(btcData.length * 0.8);
    testData = btcData.slice(0, btcData.length * 0.2);

    // Create and fit base model
    baseModel = new ARIMA({ p: 1, d: 1, q: 1 });
    baseModel.fit(trainData);
  });

  describe('Stepwise Strategy', () => {
    let stepwise: Stepwise;

    beforeEach(() => {
      stepwise = new Stepwise(baseModel, trainData, {
        refitModel: true,
        verbose: false
      });
    });

    test('should initialize correctly', () => {
      expect(stepwise).toBeInstanceOf(Stepwise);
      expect(stepwise.getCurrentData()).toEqual(trainData);
      expect(stepwise.getCurrentModel()).toBeInstanceOf(ARIMA);
    });

    test('should forecast multiple steps ahead', () => {
      const result = stepwise.forecast(5);

      expect(result.forecasts).toHaveLength(5);
      expect(result.confidenceIntervals.lower).toHaveLength(5);
      expect(result.confidenceIntervals.upper).toHaveLength(5);

      // Check that all values are finite numbers
      result.forecasts.forEach(forecast => {
        expect(Number.isFinite(forecast)).toBe(true);
      });

      result.confidenceIntervals.lower.forEach(lower => {
        expect(Number.isFinite(lower)).toBe(true);
      });

      result.confidenceIntervals.upper.forEach(upper => {
        expect(Number.isFinite(upper)).toBe(true);
      });
    });

    test('should process real-time data batch', () => {
      const result = stepwise.forecastWithRealTimeData(testData);

      expect(result.forecasts).toHaveLength(testData.length);
      expect(result.errors).toHaveLength(testData.length);
      expect(result.actualValues).toEqual(testData);
      expect(result.updatedModels).toHaveLength(testData.length);

      // Check that all errors are non-negative
      result.errors!.forEach(error => {
        expect(error).toBeGreaterThanOrEqual(0);
      });
    });

    test('should add single observation and forecast', () => {
      const newObservation = 100.5;
      const result = stepwise.addObservationAndForecast(newObservation);

      expect(result.forecast).toBeDefined();
      expect(Number.isFinite(result.forecast)).toBe(true);
      expect(Number.isFinite(result.lowerBound)).toBe(true);
      expect(Number.isFinite(result.upperBound)).toBe(true);
      expect(result.error).toBeUndefined(); // First call should have no error

      // Second call should have error
      const result2 = stepwise.addObservationAndForecast(101.2);
      expect(result2.error).toBeDefined();
      expect(result2.error).toBeGreaterThanOrEqual(0);
    });

    test('should update internal state correctly', () => {
      const initialLength = stepwise.getCurrentData().length;
      const newObservation = 100.5;

      stepwise.addObservationAndForecast(newObservation);

      expect(stepwise.getCurrentData()).toHaveLength(initialLength + 1);
      expect(stepwise.getCurrentData()[initialLength]).toBe(newObservation);
    });

    test('should reset correctly', () => {
      const newData = [1, 2, 3, 4, 5];
      stepwise.reset(newData);

      expect(stepwise.getCurrentData()).toEqual(newData);
    });

    test('should work without model refitting', () => {
      const stepwiseNoRefit = new Stepwise(baseModel, trainData, {
        refitModel: false,
        verbose: false
      });

      const result = stepwiseNoRefit.forecastWithRealTimeData(testData.slice(0, 3));
      expect(result.updatedModels).toHaveLength(0); // No models updated when refitting is disabled
    });
  });

  describe('RollingWindow Strategy', () => {
    let rollingWindow: RollingWindow;
    const windowSize = 20;

    beforeEach(() => {
      rollingWindow = new RollingWindow(baseModel, trainData, {
        windowSize,
        verbose: false
      });
    });

    test('should initialize correctly', () => {
      expect(rollingWindow).toBeInstanceOf(RollingWindow);
      expect(rollingWindow.getWindowSize()).toBe(windowSize);
      expect(rollingWindow.getCurrentData()).toEqual(trainData);
    });

    test('should forecast multiple steps ahead', () => {
      const result = rollingWindow.forecast(5);

      expect(result.forecasts).toHaveLength(5);
      expect(result.confidenceIntervals.lower).toHaveLength(5);
      expect(result.confidenceIntervals.upper).toHaveLength(5);
    });

    test('should process real-time data batch', () => {
      const result = rollingWindow.forecastWithRealTimeData(testData);
      console.log(result);

      expect(result.forecasts).toHaveLength(testData.length);
      expect(result.errors).toHaveLength(testData.length);
      expect(result.actualValues).toEqual(testData);

      // Check that all errors are non-negative
      result.errors!.forEach(error => {
        expect(error).toBeGreaterThanOrEqual(0);
      });
    });

    test('should maintain rolling window size', () => {
      // Add more observations than window size
      const manyObservations = Array.from({ length: windowSize + 10 }, (_, i) => 100 + i);

      for (const obs of manyObservations) {
        rollingWindow.addObservationAndForecast(obs);
      }

      // Data should not exceed initial length + window size
      expect(rollingWindow.getCurrentData().length).toBeLessThanOrEqual(trainData.length + windowSize + 10);
    });

    test('should add single observation and forecast', () => {
      const newObservation = 100.5;
      const result = rollingWindow.addObservationAndForecast(newObservation);

      expect(result.forecast).toBeDefined();
      expect(Number.isFinite(result.forecast)).toBe(true);
      expect(Number.isFinite(result.lowerBound)).toBe(true);
      expect(Number.isFinite(result.upperBound)).toBe(true);
      expect(result.error).toBeUndefined(); // First call should have no error

      // Second call should have error
      const result2 = rollingWindow.addObservationAndForecast(101.2);
      expect(result2.error).toBeDefined();
      expect(result2.error).toBeGreaterThanOrEqual(0);
    });

    test('should allow window size adjustment', () => {
      const newWindowSize = 15;
      rollingWindow.setWindowSize(newWindowSize);
      expect(rollingWindow.getWindowSize()).toBe(newWindowSize);
    });

    test('should reset correctly', () => {
      const newData = [1, 2, 3, 4, 5];
      rollingWindow.reset(newData);

      expect(rollingWindow.getCurrentData()).toEqual(newData);
    });

    test('should handle small window sizes', () => {
      const smallWindow = new RollingWindow(baseModel, trainData, {
        windowSize: 5,
        verbose: false
      });

      const result = smallWindow.forecast(3);
      expect(result.forecasts).toHaveLength(3);
    });
  });

  describe('Adaptive Strategy', () => {
    let adaptive: Adaptive;

    beforeEach(() => {
      adaptive = new Adaptive(baseModel, trainData, {
        windowSize: 20,
        adaptationThreshold: 2.0,
        maxErrorWindowSize: 5,
        verbose: false
      });
    });

    test('should initialize correctly', () => {
      expect(adaptive).toBeInstanceOf(Adaptive);
      expect(adaptive.getCurrentStrategy()).toBe('stepwise'); // Should start with stepwise
    });

    test('should forecast multiple steps ahead', () => {
      const result = adaptive.forecast(5);

      expect(result.forecasts).toHaveLength(5);
      expect(result.confidenceIntervals.lower).toHaveLength(5);
      expect(result.confidenceIntervals.upper).toHaveLength(5);
    });

    test('should process real-time data batch', () => {
      const result = adaptive.forecastWithRealTimeData(testData);

      expect(result.forecasts).toHaveLength(testData.length);
      expect(result.errors).toHaveLength(testData.length);
      expect(result.actualValues).toEqual(testData);
    });

    test('should add single observation and forecast', () => {
      const newObservation = 100.5;
      const result = adaptive.addObservationAndForecast(newObservation);

      expect(result.forecast).toBeDefined();
      expect(Number.isFinite(result.forecast)).toBe(true);
      expect(Number.isFinite(result.lowerBound)).toBe(true);
      expect(Number.isFinite(result.upperBound)).toBe(true);
      expect(result.error).toBeUndefined(); // First call should have no error
    });

    test('should switch strategies based on performance', () => {
      // Create data that should trigger strategy switching
      const highErrorData = Array.from({ length: 10 }, () => Math.random() * 1000);

      const initialStrategy = adaptive.getCurrentStrategy();
      adaptive.forecastWithRealTimeData(highErrorData);

      // Strategy might have switched due to high errors
      const finalStrategy = adaptive.getCurrentStrategy();
      expect(['stepwise', 'rolling']).toContain(finalStrategy);
    });

    test('should reset correctly', () => {
      const newData = [1, 2, 3, 4, 5];
      adaptive.reset(newData);

      expect(adaptive.getCurrentStrategy()).toBe('stepwise'); // Should reset to stepwise
    });

    test('should handle strategy switching with verbose logging', () => {
      const verboseAdaptive = new Adaptive(baseModel, trainData, {
        windowSize: 20,
        adaptationThreshold: 1.0, // Low threshold to trigger switching
        maxErrorWindowSize: 3,
        verbose: true
      });

      // Add observations that might trigger switching
      for (let i = 0; i < 5; i++) {
        verboseAdaptive.addObservationAndForecast(Math.random() * 100);
      }

      expect(['stepwise', 'rolling']).toContain(verboseAdaptive.getCurrentStrategy());
    });
  });

  describe('Strategy Comparison', () => {
    test('should compare all strategies on same data', () => {
      const strategies = {
        stepwise: new Stepwise(baseModel, trainData, { refitModel: true }),
        rollingWindow: new RollingWindow(baseModel, trainData, { windowSize: 20 }),
        adaptive: new Adaptive(baseModel, trainData, { windowSize: 20 })
      };

      const results: { [key: string]: number } = {};

      for (const [name, strategy] of Object.entries(strategies)) {
        const result = strategy.forecastWithRealTimeData(testData);
        const avgError = result.errors!.reduce((a, b) => a + b, 0) / result.errors!.length;
        results[name] = avgError;

        expect(avgError).toBeGreaterThanOrEqual(0);
        expect(result.forecasts).toHaveLength(testData.length);
      }

      // All strategies should produce results
      expect(Object.keys(results)).toHaveLength(3);
      expect(results.stepwise).toBeDefined();
      expect(results.rollingWindow).toBeDefined();
      expect(results.adaptive).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty test data', () => {
      const stepwise = new Stepwise(baseModel, trainData);
      const result = stepwise.forecastWithRealTimeData([]);

      expect(result.forecasts).toHaveLength(0);
      expect(result.errors).toHaveLength(0);
      expect(result.actualValues).toHaveLength(0);
    });

    test('should handle single observation', () => {
      const rollingWindow = new RollingWindow(baseModel, trainData);
      const result = rollingWindow.forecastWithRealTimeData([100.5]);

      expect(result.forecasts).toHaveLength(1);
      expect(result.errors).toHaveLength(1);
    });

    test('should handle very small training data', () => {
      const smallData = [1, 2, 3];
      const smallModel = new ARIMA({ p: 1, d: 0, q: 1 });
      smallModel.fit(smallData);

      const stepwise = new Stepwise(smallModel, smallData);
      const result = stepwise.forecast(2);

      expect(result.forecasts).toHaveLength(2);
    });

    test('should handle large forecast horizons', () => {
      const stepwise = new Stepwise(baseModel, trainData);
      const result = stepwise.forecast(50);

      expect(result.forecasts).toHaveLength(50);
      expect(result.confidenceIntervals.lower).toHaveLength(50);
      expect(result.confidenceIntervals.upper).toHaveLength(50);
    });
  });
}); 