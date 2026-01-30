
"use client"

import { useState } from "react"
import { Navbar } from "@/components/Navbar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import api from "@/lib/api"
import { UploadCloud, FileSpreadsheet, Download } from "lucide-react"

export default function BulkPredictionPage() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
        setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError("")
    setResult(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await api.post("/predict/bulk", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      })
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || "Upload failed. Please check the CSV format.")
    } finally {
      setLoading(false)
    }
  }

  const downloadSample = () => {
    // Create a dummy CSV for download
    const csvContent = "age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active\n50,2,168,72,120,80,1,1,0,0,1\n60,1,160,80,140,90,2,1,0,0,1"
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.setAttribute("href", url)
    link.setAttribute("download", "sample_cardio_data.csv")
    document.body.appendChild(link)
    link.click()
  }

  return (
    <div className="min-h-screen bg-muted/20 pb-12">
      <Navbar />
      <div className="container px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Bulk Analysis</h1>

        <div className="grid gap-8 lg:grid-cols-3">
             <Card className="lg:col-span-1 h-fit">
                <CardHeader>
                    <CardTitle>Upload CSV</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 flex flex-col items-center justify-center text-center">
                        <UploadCloud className="h-10 w-10 text-muted-foreground mb-4" />
                        <Input type="file" accept=".csv" onChange={handleFileChange} className="max-w-xs" />
                        <p className="text-xs text-muted-foreground mt-2">Only .csv files supported</p>
                    </div>
                    
                    <Button onClick={handleUpload} className="w-full" disabled={!file || loading}>
                        {loading ? "Processing..." : "Analyze Dataset"}
                    </Button>
                    
                    {error && <p className="text-red-500 text-sm font-medium">{error}</p>}

                    <div className="pt-4 border-t">
                        <Button variant="ghost" size="sm" onClick={downloadSample} className="w-full text-muted-foreground">
                            <Download className="mr-2 h-4 w-4" /> Download Sample Template
                        </Button>
                    </div>
                </CardContent>
             </Card>

             <Card className="lg:col-span-2">
                <CardHeader>
                    <CardTitle>Results Overview</CardTitle>
                </CardHeader>
                <CardContent>
                    {!result ? (
                        <div className="flex flex-col items-center justify-center h-64 text-muted-foreground bg-muted/10 rounded-lg">
                            <FileSpreadsheet className="h-12 w-12 mb-4 opacity-20" />
                            <p>Upload a file to see analysis results.</p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                                <div className="bg-primary/10 p-4 rounded-lg">
                                    <div className="text-sm font-medium text-muted-foreground">Total Records</div>
                                    <div className="text-2xl font-bold">{result.total_records}</div>
                                </div>
                                <div className="bg-orange-100 dark:bg-orange-900/20 p-4 rounded-lg">
                                    <div className="text-sm font-medium text-muted-foreground">Average Risk</div>
                                    <div className="text-2xl font-bold">{(result.avg_risk * 100).toFixed(1)}%</div>
                                </div>
                            </div>

                            <div className="border rounded-md overflow-x-auto">
                                <table className="w-full text-sm text-left">
                                    <thead className="bg-muted text-muted-foreground font-medium">
                                        <tr>
                                            <th className="p-3">ID</th>
                                            <th className="p-3">Age</th>
                                            <th className="p-3">Gender</th>
                                            <th className="p-3">BP</th>
                                            <th className="p-3">BMI Risk</th>
                                            <th className="p-3">Pred. Probability</th>
                                            <th className="p-3">Risk Level</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y">
                                        {result.predictions.slice(0, 10).map((row: any, idx: number) => (
                                            <tr key={idx} className="hover:bg-muted/50">
                                                <td className="p-3">{idx + 1}</td>
                                                <td className="p-3">{(row.age / 365.25).toFixed(0)}</td>
                                                <td className="p-3">{row.gender === 1 ? 'F' : 'M'}</td>
                                                <td className="p-3">{row.ap_hi}/{row.ap_lo}</td>
                                                <td className="p-3">{row.weight / ((row.height/100)**2) > 25 ? 'High' : 'Normal'}</td>
                                                <td className="p-3">{(row.Prediction_Probability * 100).toFixed(1)}%</td>
                                                <td className={`p-3 font-medium ${row.Predicted_Cardio_Disease === 'Yes' ? 'text-red-500' : 'text-green-600'}`}>
                                                    {row.Predicted_Cardio_Disease === 'Yes' ? 'High' : 'Low'}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            {result.predictions.length > 10 && (
                                <p className="text-center text-xs text-muted-foreground">Showing first 10 records only.</p>
                            )}
                        </div>
                    )}
                </CardContent>
             </Card>
        </div>
      </div>
    </div>
  )
}
