import { ARIMA } from '../../models/arima';
import { TimeSeriesExample } from '../../utils/example';

describe('ARIMA Model', () => {
  test('should fit and forecast correctly', () => {
    const data = TimeSeriesExample.generateSampleData(50);
    const model = new ARIMA({ p: 1, d: 1, q: 1 });

    const fitResult = model.fit(data);
    expect(fitResult.coefficients.ar).toBeDefined();
    expect(fitResult.aic).toBeGreaterThan(0);

    const forecast = model.forecast(5);
    expect(forecast.forecast).toHaveLength(5);
    expect(forecast.lowerBound).toHaveLength(5);
    expect(forecast.upperBound).toHaveLength(5);
  });
});