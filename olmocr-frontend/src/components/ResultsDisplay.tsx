import React from 'react';
import './ResultsDisplay.css';

interface ResultsDisplayProps {
  data: any;
  loading: boolean;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ data, loading }) => {
  if (loading) return null;
  if (!data) return null;

  return (
    <div className="results-container">
      <h2>Extracted Data</h2>
      {data.entities && (
        <div className="entities-section">
          <h3>Entities</h3>
          <div className="entity-list">
            <div>
              <strong>Names:</strong> {data.entities.names?.join(', ') || 'None'}
            </div>
            <div>
              <strong>Dates:</strong> {data.entities.dates?.join(', ') || 'None'}
            </div>
            <div>
              <strong>Addresses:</strong> {data.entities.addresses?.join(', ') || 'None'}
            </div>
          </div>
        </div>
      )}
      {data.tables && data.tables.length > 0 && (
        <div className="tables-section">
          <h3>Tables</h3>
          {data.tables.map((table: any, idx: number) => (
            <table className="extracted-table" key={idx}>
              <thead>
                <tr>
                  {table.headers.map((header: string, i: number) => (
                    <th key={i}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row: string[], i: number) => (
                  <tr key={i}>
                    {row.map((cell: string, j: number) => (
                      <td key={j}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
        </div>
      )}
      {data.raw_text && (
        <div className="raw-text-section">
          <h3>Raw Text</h3>
          <pre className="raw-text-block">{data.raw_text}</pre>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay; 