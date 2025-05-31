import { ARIMA } from "../models/arima";
import { Diagnostics } from "./diagnostics";
import { AutoARIMA } from "./model-selection";
import { Stepwise, RollingWindow, Adaptive } from "./strategies";

export class TimeSeriesExample {
  static generateSampleData(n: number = 100): number[] {
    const data: number[] = [];
    let value = 100;

    for (let i = 0; i < n; i++) {

      const trend = 0.1 * i;
      const seasonal = 10 * Math.sin(2 * Math.PI * i / 12);
      const noise = (Math.random() - 0.5) * 5;

      value += trend + seasonal + noise;
      data.push(value);
    }

    return data;
  }

  static runBasicExample(): void {
    console.log('=== Basic ARIMA Example ===');
    const data = this.generateSampleData(100);

    console.log('Generated sample time series data');
    console.log('Data length:', data.length);
    console.log('First 10 values:', data.slice(0, 10));

    const autoResult = AutoARIMA.findBestARIMA(data, 3, 2, 3);
    console.log('Best ARIMA model:', autoResult.bestParams);
    console.log('Best AIC score:', autoResult.bestScore);

    const arima = new ARIMA(autoResult.bestParams);
    const fitResult = arima.fit(data);

    console.log('Model coefficients:', fitResult.coefficients);
    console.log('AIC:', fitResult.aic);
    console.log('BIC:', fitResult.bic);

    const forecast = arima.forecast(10);
    console.log('10-step forecast:', forecast.forecast);
    console.log('Confidence intervals:', {
      lower: forecast.lowerBound,
      upper: forecast.upperBound
    });

    const diagnostics = Diagnostics.plotResiduals(fitResult.residuals);
    console.log('Residuals diagnostics:', diagnostics);
  }

  static runStrategiesExample(): void {
    console.log('\n=== Real-Time Forecasting Strategies Example ===');

    // 80% random data for training, 20% for testing, 20% for real-time simulation
    const fullData = this.generateSampleData(120);
    const trainData = fullData.slice(0, 80);
    const testData = fullData.slice(80, 100);
    const realTimeData = fullData.slice(100);

    console.log(`Training data: ${trainData.length} points`);
    console.log(`Test data: ${testData.length} points`);
    console.log(`Real-time data: ${realTimeData.length} points`);

    // Find best ARIMA model
    const autoResult = AutoARIMA.findBestARIMA(trainData, 3, 2, 3);
    const baseModel = new ARIMA(autoResult.bestParams);
    baseModel.fit(trainData);

    console.log('\nBase model fitted with parameters:', autoResult.bestParams);

    // Test all three strategies
    this.testStepwiseStrategy(baseModel, trainData, testData, realTimeData);
    this.testRollingWindowStrategy(baseModel, trainData, testData, realTimeData);
    this.testAdaptiveStrategy(baseModel, trainData, testData, realTimeData);
  }

  private static testStepwiseStrategy(
    baseModel: ARIMA,
    trainData: number[],
    testData: number[],
    realTimeData: number[]
  ): void {
    console.log('\n--- Stepwise Strategy ---');

    const stepwise = new Stepwise(baseModel, trainData, {
      refitModel: true,
      verbose: false
    });

    console.log('Batch processing test data...');
    const batchResult = stepwise.forecastWithRealTimeData(testData);
    const avgError = batchResult.errors!.reduce((a, b) => a + b, 0) / batchResult.errors!.length;
    console.log(`Average error on test data: ${avgError.toFixed(3)}`);

    console.log('Real-time simulation:');
    for (let i = 0; i < Math.min(5, realTimeData.length); i++) {
      const result = stepwise.addObservationAndForecast(realTimeData[i]);
      console.log(`  Observation ${i + 1}: ${realTimeData[i].toFixed(3)} -> Next forecast: ${result.forecast.toFixed(3)}${result.error ? `, Error: ${result.error.toFixed(3)}` : ''}`);
    }
  }

  private static testRollingWindowStrategy(
    baseModel: ARIMA,
    trainData: number[],
    testData: number[],
    realTimeData: number[]
  ): void {
    console.log('\n--- Rolling Window Strategy ---');

    const rollingWindow = new RollingWindow(baseModel, trainData, {
      windowSize: 30,
      verbose: false
    });

    console.log('Batch processing test data...');
    const batchResult = rollingWindow.forecastWithRealTimeData(testData);
    const avgError = batchResult.errors!.reduce((a, b) => a + b, 0) / batchResult.errors!.length;
    console.log(`Average error on test data: ${avgError.toFixed(3)}`);
    console.log(`Window size: ${rollingWindow.getWindowSize()}`);

    console.log('Real-time simulation:');
    for (let i = 0; i < Math.min(5, realTimeData.length); i++) {
      const result = rollingWindow.addObservationAndForecast(realTimeData[i]);
      console.log(`  Observation ${i + 1}: ${realTimeData[i].toFixed(3)} -> Next forecast: ${result.forecast.toFixed(3)}${result.error ? `, Error: ${result.error.toFixed(3)}` : ''}`);
    }
  }

  private static testAdaptiveStrategy(
    baseModel: ARIMA,
    trainData: number[],
    testData: number[],
    realTimeData: number[]
  ): void {
    console.log('\n--- Adaptive Strategy ---');

    const adaptive = new Adaptive(baseModel, trainData, {
      windowSize: 30,
      adaptationThreshold: 3.0,
      maxErrorWindowSize: 10,
      verbose: false
    });

    console.log('Batch processing test data...');
    const batchResult = adaptive.forecastWithRealTimeData(testData);
    const avgError = batchResult.errors!.reduce((a, b) => a + b, 0) / batchResult.errors!.length;
    console.log(`Average error on test data: ${avgError.toFixed(3)}`);
    console.log(`Current strategy: ${adaptive.getCurrentStrategy()}`);

    console.log('Real-time simulation:');
    for (let i = 0; i < Math.min(10, realTimeData.length); i++) {
      const result = adaptive.addObservationAndForecast(realTimeData[i]);
      console.log(`  Observation ${i + 1}: ${realTimeData[i].toFixed(3)} -> Next forecast: ${result.forecast.toFixed(3)}${result.error ? `, Error: ${result.error.toFixed(3)}` : ''} [${adaptive.getCurrentStrategy()}]`);
    }
  }

  static runComparisonExample(): void {
    console.log('\n=== Strategy Comparison Example ===');

    const fullData = this.generateSampleData(150);
    const trainData = fullData.slice(0, 100);
    const testData = fullData.slice(100);

    const autoResult = AutoARIMA.findBestARIMA(trainData, 3, 2, 3);
    const baseModel = new ARIMA(autoResult.bestParams);
    baseModel.fit(trainData);

    const strategies = {
      stepwise: new Stepwise(baseModel, trainData, { refitModel: true }),
      rollingWindow: new RollingWindow(baseModel, trainData, { windowSize: 40 }),
      adaptive: new Adaptive(baseModel, trainData, {
        windowSize: 40,
        adaptationThreshold: 2.5
      })
    };

    console.log('Comparing strategies on test data...');
    const results: { [key: string]: number } = {};

    for (const [name, strategy] of Object.entries(strategies)) {
      const result = strategy.forecastWithRealTimeData(testData);
      const avgError = result.errors!.reduce((a, b) => a + b, 0) / result.errors!.length;
      results[name] = avgError;
      console.log(`${name}: Average error = ${avgError.toFixed(3)}`);
    }

    const bestStrategy = Object.entries(results).reduce((a, b) =>
      results[a[0]] < results[b[0]] ? a : b
    );
    console.log(`\nBest performing strategy: ${bestStrategy[0]} (error: ${bestStrategy[1].toFixed(3)})`);
  }

  static runFullExample(): void {
    console.log('🚀 Running Time Series Forecasting Examples\n');

    try {
      this.runBasicExample();
      this.runStrategiesExample();
      this.runComparisonExample();

      console.log('\n✅ All examples completed successfully!');
    } catch (error) {
      console.error('❌ Error running examples:', error);
    }
  }
}