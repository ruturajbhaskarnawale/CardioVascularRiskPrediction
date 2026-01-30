
"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Navbar } from "@/components/Navbar"
import api from "@/lib/api"

export default function SignupPage() {
  const router = useRouter()
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await api.post("/auth/signup", { username, password })
      setSuccess("Account created! Redirecting to login...")
      setTimeout(() => router.push("/login"), 2000)
    } catch (err: any) {
      setError(err.response?.data?.detail || "Signup failed")
    }
  }

  return (
    <div className="min-h-screen bg-muted/50">
      <Navbar />
      <div className="flex items-center justify-center py-12">
        <Card className="w-[350px]">
          <CardHeader>
            <CardTitle>Sign Up</CardTitle>
            <CardDescription>Create a new account to get started.</CardDescription>
          </CardHeader>
          <form onSubmit={handleSignup}>
            <CardContent>
              <div className="grid w-full items-center gap-4">
                <div className="flex flex-col space-y-1.5">
                  <Input 
                    type="text" 
                    placeholder="Choose a Username" 
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                  />
                </div>
                <div className="flex flex-col space-y-1.5">
                  <Input 
                    type="password" 
                    placeholder="Choose a Password" 
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
                {error && <p className="text-sm text-destructive">{error}</p>}
                {success && <p className="text-sm text-green-600">{success}</p>}
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
               <Button variant="outline" asChild>
                <Link href="/login">Back to Login</Link>
              </Button>
              <Button type="submit">Sign Up</Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </div>
  )
}
