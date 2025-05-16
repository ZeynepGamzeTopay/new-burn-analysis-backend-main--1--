    using System;
    using System.ComponentModel.DataAnnotations;
    using System.ComponentModel.DataAnnotations.Schema;

    namespace BurnAnalysisApp.Models
    {
        public class Visit
        {
            [Key]
            public int VisitID { get; set; }

            // PatientInfo ile ilişkilendirme (Foreign Key)
            [Required]
            public int PatientID { get; set; }

            [ForeignKey("PatientID")]
            public PatientInfo? Patient { get; set; } 

            // Ziyaret bilgileri
            [Required]
            public DateTime VisitDate { get; set; } // Hastaneye geliş tarihi

            public string? PhotoPath { get; set; } // Ziyaret sırasında çekilen yeni fotoğraf

            public string? LabResultsFilePath { get; set; } // Kan tahlili dosyası (PDF, JPG vb.)

            public string? PrescribedMedications { get; set; } // Yazılan ilaçlar (Örn: "Parol, Amoklavin")

            public string? Notes { get; set; } // Ekstra notlar (Doktorun açıklamaları)
        }
    }
