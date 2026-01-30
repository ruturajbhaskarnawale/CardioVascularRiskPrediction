
"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Navbar } from "@/components/Navbar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity, FileSpreadsheet, MapPin, GraduationCap, ArrowRight } from "lucide-react"

export default function Dashboard() {
  const [username, setUsername] = useState<string | null>(null)

  useEffect(() => {
    // Check for user in localStorage (simulating auth state)
    const user = localStorage.getItem("username")
    setUsername(user)
  }, [])

  return (
    <div className="min-h-screen bg-muted/20">
      <Navbar />
      <main className="container px-4 py-8">
        <div className="mb-8">
           <h1 className="text-3xl font-bold tracking-tight">
            {username ? `Welcome back, ${username}` : "CardioHealth Dashboard"}
          </h1>
          <p className="text-muted-foreground mt-2">
            Your centralized platform for cardiovascular risk assessment and health management.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <Card className="hover:shadow-lg transition-shadow cursor-pointer border-t-4 border-t-primary">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Single Prediction</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Assess Risk</div>
              <p className="text-xs text-muted-foreground">
                Get a personalized risk profile based on your health metrics.
              </p>
              <Link href="/predict" className="mt-4 block">
                 <Button className="w-full" size="sm">Start Assessment <ArrowRight className="ml-2 h-4 w-4"/></Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow cursor-pointer border-t-4 border-t-blue-500">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Bulk Analysis</CardTitle>
              <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Upload Data</div>
              <p className="text-xs text-muted-foreground">
                Process multiple patient records via CSV upload.
              </p>
               <Link href="/bulk" className="mt-4 block">
                 <Button variant="outline" className="w-full" size="sm">Upload CSV</Button>
              </Link>
            </CardContent>
          </Card>

           <Card className="hover:shadow-lg transition-shadow cursor-pointer border-t-4 border-t-green-500">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Local Resources</CardTitle>
              <MapPin className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Find Care</div>
              <p className="text-xs text-muted-foreground">
                Locate hospitals and clinics near you in Maharashtra.
              </p>
               <Link href="/resources" className="mt-4 block">
                 <Button variant="outline" className="w-full" size="sm">Find Resources</Button>
              </Link>
            </CardContent>
          </Card>

           <Card className="hover:shadow-lg transition-shadow cursor-pointer border-t-4 border-t-orange-500">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Education Hub</CardTitle>
              <GraduationCap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Learn More</div>
              <p className="text-xs text-muted-foreground">
                Explore videos and articles on heart health.
              </p>
               <Link href="/education" className="mt-4 block">
                 <Button variant="outline" className="w-full" size="sm">Browse Content</Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* Recent History or Stats could go here */}
      </main>
    </div>
  )
}
