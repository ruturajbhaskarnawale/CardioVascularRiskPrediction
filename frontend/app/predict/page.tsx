
"use client"

import { useState } from "react"
import { Navbar } from "@/components/Navbar"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/simple-select"
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import api from "@/lib/api"
import { AlertCircle, CheckCircle2 } from "lucide-react"

export default function SinglePredictionPage() {
  const [formData, setFormData] = useState({
    full_name: "", phone_number: "",
    age: 50, height: 165, weight: 70,
    ap_hi: 120, ap_lo: 80, gender: "Male",
    cholesterol: 1, gluc: 1, smoke: 0, alco: 0, active: 1, stress: 1
  })
  
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleChange = (e: any) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "stress"].includes(name) 
        ? Number(value) 
        : value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError("")
    setResult(null)

    try {
      const username = localStorage.getItem("username") || "guest"
      const response = await api.post("/predict/", formData, {
        params: { username, save_history: true }
      })
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || "Prediction failed")
    } finally {
      setLoading(false)
    }
  }

  const downloadPDF = async () => {
    try {
        const response = await api.post("/predict/report/pdf", formData, {
            responseType: 'blob'
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'CardioReport.pdf');
        document.body.appendChild(link);
        link.click();
    } catch (err) {
        alert("Failed to download PDF");
    }
  }

  return (
    <div className="min-h-screen bg-muted/20 pb-12">
      <Navbar />
      <div className="container px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Cardiovascular Risk Assessment</h1>
        
        <div className="grid gap-8 lg:grid-cols-3">
            {/* Input Form */}
            <Card className="lg:col-span-2">
                <CardHeader>
                    <CardTitle>Patient Details</CardTitle>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label htmlFor="full_name">Full Name</Label>
                                <Input name="full_name" value={formData.full_name} onChange={handleChange} required />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="phone_number">Phone</Label>
                                <Input name="phone_number" value={formData.phone_number} onChange={handleChange} />
                            </div>
                            
                            <div className="space-y-2">
                                <Label htmlFor="age">Age (Years)</Label>
                                <Input type="number" name="age" value={formData.age} onChange={handleChange} required />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="gender">Gender</Label>
                                <Select name="gender" value={formData.gender} onChange={handleChange}>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                </Select>
                            </div>
                            
                            <div className="space-y-2">
                                <Label htmlFor="height">Height (cm)</Label>
                                <Input type="number" name="height" value={formData.height} onChange={handleChange} required />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="weight">Weight (kg)</Label>
                                <Input type="number" name="weight" value={formData.weight} onChange={handleChange} required />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="ap_hi">Systolic BP</Label>
                                <Input type="number" name="ap_hi" value={formData.ap_hi} onChange={handleChange} required />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="ap_lo">Diastolic BP</Label>
                                <Input type="number" name="ap_lo" value={formData.ap_lo} onChange={handleChange} required />
                            </div>

                             <div className="space-y-2">
                                <Label htmlFor="cholesterol">Cholesterol</Label>
                                <Select name="cholesterol" value={formData.cholesterol} onChange={handleChange}>
                                    <option value={1}>Normal</option>
                                    <option value={2}>Above Normal</option>
                                    <option value={3}>Well Above Normal</option>
                                </Select>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="gluc">Glucose</Label>
                                <Select name="gluc" value={formData.gluc} onChange={handleChange}>
                                    <option value={1}>Normal</option>
                                    <option value={2}>Above Normal</option>
                                    <option value={3}>Well Above Normal</option>
                                </Select>
                            </div>

                             <div className="space-y-2">
                                <Label htmlFor="smoke">Smoker</Label>
                                <Select name="smoke" value={formData.smoke} onChange={handleChange}>
                                    <option value={0}>No</option>
                                    <option value={1}>Yes</option>
                                </Select>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="alco">Alcohol Intake</Label>
                                <Select name="alco" value={formData.alco} onChange={handleChange}>
                                    <option value={0}>No</option>
                                    <option value={1}>Moderate</option>
                                    <option value={2}>Heavy</option>
                                </Select>
                            </div>
                             <div className="space-y-2">
                                <Label htmlFor="active">Physical Activity</Label>
                                <Select name="active" value={formData.active} onChange={handleChange}>
                                    <option value={0}>Sedentary</option>
                                    <option value={1}>Active</option>
                                </Select>
                            </div>
                        </div>

                        {error && <p className="text-red-500 text-sm">{error}</p>}
                        <Button type="submit" className="w-full" disabled={loading}>
                            {loading ? "Analyzing..." : "Predict Risk"}
                        </Button>
                    </form>
                </CardContent>
            </Card>

            {/* Results Section */}
            {result && (
                <div className="lg:col-span-1 space-y-6">
                    <Card className={`border-t-4 ${result.risk_color === 'red' ? 'border-destructive' : 'border-green-500'}`}>
                        <CardHeader>
                            <CardTitle>Assessment Result</CardTitle>
                        </CardHeader>
                        <CardContent className="text-center">
                            <div className={`text-3xl font-bold mb-2 ${result.risk_color === 'red' ? 'text-destructive' : 'text-green-600'}`}>
                                {result.risk_level}
                            </div>
                            <div className="text-4xl font-extrabold mb-4">
                                {(result.probability * 100).toFixed(1)}%
                            </div>
                            <p className="text-sm text-muted-foreground mb-4">Probability of cardiovascular disease.</p>
                             <Button onClick={downloadPDF} variant="outline" className="w-full">
                                Download Official Report (PDF)
                            </Button>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base">Key Risk Factors</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="prose prose-sm dark:prose-invert">
                                <div dangerouslySetInnerHTML={{ __html: result.factors.replace(/\n/g, '<br/>').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>') }} />
                            </div>
                        </CardContent>
                    </Card>

                     <Card>
                        <CardHeader>
                            <CardTitle className="text-base">Recommendations</CardTitle>
                        </CardHeader>
                        <CardContent>
                             <div className="prose prose-sm dark:prose-invert">
                                <div dangerouslySetInnerHTML={{ __html: result.recommendations.replace(/\n/g, '<br/>').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>') }} />
                            </div>
                        </CardContent>
                    </Card>
                </div>
            )}
        </div>
      </div>
    </div>
  )
}
