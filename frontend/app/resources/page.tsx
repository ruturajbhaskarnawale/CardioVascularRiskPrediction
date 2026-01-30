
"use client"

import { useState } from "react"
import { Navbar } from "@/components/Navbar"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/simple-select"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import api from "@/lib/api"
import { MapPin, Search } from "lucide-react"

export default function ResourcesPage() {
  const [city, setCity] = useState("Mumbai")
  const [resourceType, setResourceType] = useState("Hospital")
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setSearched(true)
    try {
      const response = await api.get("/resources/", {
        params: { city, resource_type: resourceType }
      })
      setResults(response.data)
    } catch (error) {
      console.error("Search failed", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-muted/20 pb-12">
      <Navbar />
      <div className="container px-4 py-8">
        <h1 className="text-3xl font-bold mb-2">Local Healthcare Resources</h1>
        <p className="text-muted-foreground mb-8">Find hospitals and clinics near you in Maharashtra.</p>

        <Card className="mb-8">
             <CardContent className="pt-6">
                <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4 items-end">
                    <div className="flex-1 space-y-2 w-full">
                        <label className="text-sm font-medium">Select City</label>
                         <Select value={city} onChange={(e) => setCity(e.target.value)}>
                            <option value="Mumbai">Mumbai</option>
                             <option value="Pune">Pune</option>
                             <option value="Nagpur">Nagpur</option>
                             <option value="Nashik">Nashik</option>
                             <option value="Aurangabad">Aurangabad</option>
                        </Select>
                    </div>
                     <div className="flex-1 space-y-2 w-full">
                        <label className="text-sm font-medium">Resource Type</label>
                         <Select value={resourceType} onChange={(e) => setResourceType(e.target.value)}>
                             <option value="Hospital">Hospital</option>
                             <option value="Clinic">Clinic</option>
                             <option value="Pharmacy">Pharmacy</option>
                        </Select>
                    </div>
                    <Button type="submit" disabled={loading} className="w-full md:w-auto">
                        <Search className="mr-2 h-4 w-4" />
                        {loading ? "Searching..." : "Find Resources"}
                    </Button>
                </form>
             </CardContent>
        </Card>

        <div className="space-y-4">
             {searched && results.length === 0 && !loading && (
                 <div className="text-center py-12 text-muted-foreground">
                     <MapPin className="h-12 w-12 mx-auto mb-4 opacity-20" />
                     No resources found matching your criteria.
                 </div>
             )}

             <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                 {results.map((item, idx) => (
                     <Card key={idx} className="hover:border-primary/50 transition-colors">
                         <CardHeader>
                             <CardTitle className="text-lg">{item.Facility_Name}</CardTitle>
                             <CardDescription>{item.Facility_Type}</CardDescription>
                         </CardHeader>
                         <CardContent>
                             <div className="text-sm space-y-2">
                                 <div className="flex items-start">
                                     <MapPin className="h-4 w-4 mr-2 mt-0.5 text-muted-foreground shrink-0" />
                                     <span>{item.Facility_Address || "Address not available"}</span>
                                 </div>
                                 <div className="font-medium text-xs bg-primary/10 text-primary w-fit px-2 py-1 rounded">
                                     {item.distance_km} km away
                                 </div>
                             </div>
                         </CardContent>
                     </Card>
                 ))}
             </div>
        </div>
      </div>
    </div>
  )
}
